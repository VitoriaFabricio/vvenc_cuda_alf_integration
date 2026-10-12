/* -----------------------------------------------------------------------------
AlfGpuCuda.cu - CUDA backend for the VVenC ALF luma reconstruction

Per worker (encoder thread) it owns: a CUDA stream, pinned staging buffers, device
buffers and a device coefficient table (16 fixed sets uploaded once + APS sets of
the current picture uploaded per call). Every call is timed with CUDA events:
  H2D (band + per-CTU set map + APS + optional classes) | classify | filter | D2H
plus host-side pack/unpack and wall clock. GPU energy is read through NVML.
------------------------------------------------------------------------------- */

#include "../AlfGpu.h"
#include "AlfGpuKernels.cuh"

#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

namespace alfgpu
{

#define ALF_CUDA_CHECK( x )                                                                       \
  do {                                                                                            \
    cudaError_t e_ = ( x );                                                                       \
    if( e_ != cudaSuccess ) {                                                                     \
      fprintf( stderr, "[AlfGpu] CUDA error %s at %s:%d\n", cudaGetErrorString( e_ ), __FILE__, __LINE__ ); \
      abort();                                                                                    \
    }                                                                                             \
  } while( 0 )

class CudaBackend : public Backend
{
  struct Worker
  {
    cudaStream_t stream = nullptr;
    cudaEvent_t  ev[5];
    size_t capIn = 0, capOut = 0, capCls = 0, capMap = 0;
    Pel     *hIn = nullptr, *hOut = nullptr, *dIn = nullptr, *dOut = nullptr;
    uint8_t *hCls = nullptr, *dCls = nullptr;
    int16_t *hMap = nullptr, *dMap = nullptr;
    int16_t *hAps = nullptr;                      // pinned: [coef | clip] of APS sets
    int16_t *dCoef = nullptr, *dClip = nullptr;   // kMaxSets * kSetStride each

    ~Worker()
    {
      if( !stream ) return;
      cudaStreamSynchronize( stream );
      for( auto& e : ev ) cudaEventDestroy( e );
      cudaFreeHost( hIn ); cudaFreeHost( hOut ); cudaFreeHost( hCls ); cudaFreeHost( hMap ); cudaFreeHost( hAps );
      cudaFree( dIn ); cudaFree( dOut ); cudaFree( dCls ); cudaFree( dMap ); cudaFree( dCoef ); cudaFree( dClip );
      cudaStreamDestroy( stream );
    }
  };

  int m_device = 0;
  std::vector<int16_t> m_fixedCoef, m_fixedClip;
  std::vector<std::unique_ptr<Worker>> m_workers;
  std::mutex m_mutex;
  bool m_nvmlOk = false;
  nvmlDevice_t m_nvmlDev{};

  template<typename T> static void growHost( T*& p, size_t& cap, size_t n )
  {
    if( n <= cap ) return;
    if( p ) ALF_CUDA_CHECK( cudaFreeHost( p ) );
    ALF_CUDA_CHECK( cudaMallocHost( ( void** ) &p, n * sizeof( T ) ) );
    cap = n;
  }
  template<typename T> static void growDev( T*& p, size_t n, size_t oldCap )
  {
    if( n <= oldCap && p ) return;
    if( p ) ALF_CUDA_CHECK( cudaFree( p ) );
    ALF_CUDA_CHECK( cudaMalloc( ( void** ) &p, n * sizeof( T ) ) );
  }

  Worker* getWorker( int idx )
  {
    std::lock_guard<std::mutex> lock( m_mutex );
    if( idx >= ( int ) m_workers.size() ) m_workers.resize( idx + 1 );
    if( !m_workers[idx] )
    {
      ALF_CUDA_CHECK( cudaSetDevice( m_device ) );
      Worker* w = new Worker;
      ALF_CUDA_CHECK( cudaStreamCreateWithFlags( &w->stream, cudaStreamNonBlocking ) );
      for( auto& e : w->ev ) ALF_CUDA_CHECK( cudaEventCreate( &e ) );
      const size_t tab = ( size_t ) kMaxSets * kSetStride;
      ALF_CUDA_CHECK( cudaMalloc( ( void** ) &w->dCoef, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMalloc( ( void** ) &w->dClip, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMemset( w->dCoef, 0, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMemset( w->dClip, 0, tab * sizeof( int16_t ) ) );
      ALF_CUDA_CHECK( cudaMemcpy( w->dCoef, m_fixedCoef.data(), m_fixedCoef.size() * sizeof( int16_t ), cudaMemcpyHostToDevice ) );
      ALF_CUDA_CHECK( cudaMemcpy( w->dClip, m_fixedClip.data(), m_fixedClip.size() * sizeof( int16_t ), cudaMemcpyHostToDevice ) );
      ALF_CUDA_CHECK( cudaMallocHost( ( void** ) &w->hAps, 2 * kMaxAps * kSetStride * sizeof( int16_t ) ) );
      m_workers[idx].reset( w );
    }
    return m_workers[idx].get();
  }

public:
  ~CudaBackend() override
  {
    m_workers.clear();
    if( m_nvmlOk ) nvmlShutdown();
  }

  const char* name() const override { return "cuda"; }

  bool init( const int16_t* fixedCoef, const int16_t* fixedClip, int, int, int numWorkers ) override
  {
    int n = 0;
    if( cudaGetDeviceCount( &n ) != cudaSuccess || n == 0 ) return false;
    if( const char* d = getenv( "VVENC_ALF_GPU_DEVICE" ) ) m_device = std::max( 0, std::min( n - 1, atoi( d ) ) );
    ALF_CUDA_CHECK( cudaSetDevice( m_device ) );
    cudaDeviceProp prop;
    ALF_CUDA_CHECK( cudaGetDeviceProperties( &prop, m_device ) );
    fprintf( stderr, "[AlfGpu] CUDA device %d: %s (sm_%d%d)\n", m_device, prop.name, prop.major, prop.minor );

    m_fixedCoef.assign( fixedCoef, fixedCoef + kNumFixedSets * kSetStride );
    m_fixedClip.assign( fixedClip, fixedClip + kNumFixedSets * kSetStride );
    m_workers.resize( std::max( 1, numWorkers ) );

    if( nvmlInit_v2() == NVML_SUCCESS )
    {
      char pci[32];
      if( cudaDeviceGetPCIBusId( pci, sizeof( pci ), m_device ) == cudaSuccess &&
          nvmlDeviceGetHandleByPciBusId_v2( pci, &m_nvmlDev ) == NVML_SUCCESS )
      {
        m_nvmlOk = true;
      }
    }
    return true;
  }

  bool readEnergy_mJ( unsigned long long& e ) override
  {
    return m_nvmlOk && nvmlDeviceGetTotalEnergyConsumption( m_nvmlDev, &e ) == NVML_SUCCESS;
  }
  bool readPower_mW( unsigned& p ) override
  {
    return m_nvmlOk && nvmlDeviceGetPowerUsage( m_nvmlDev, &p ) == NVML_SUCCESS;
  }

  void filterLumaBand( const BandParams& p, int mode, int worker, Timing& t ) override
  {
    using clk = std::chrono::steady_clock;
    auto ms   = []( clk::time_point a, clk::time_point b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); };
    const auto t0 = clk::now();

    Worker* w = getWorker( worker );
    ALF_CUDA_CHECK( cudaSetDevice( m_device ) );

    const int    inPitch   = p.picWidth + 2 * kHalo;
    const size_t inN       = ( size_t ) inPitch * ( p.bandH + 2 * kHalo );
    const size_t outN      = ( size_t ) p.picWidth * p.bandH;
    const int    clsStride = p.picWidth >> 2;
    const size_t clsN      = ( size_t ) clsStride * ( ( p.bandH + 3 ) >> 2 );
    const int    ctuRows   = ( p.bandH + p.ctuSize - 1 ) / p.ctuSize;
    const size_t mapN      = ( size_t ) ctuRows * p.numCtuCols;

    { size_t c = w->capIn;  growHost( w->hIn,  w->capIn,  inN  ); growDev( w->dIn,  inN,  c ); }
    { size_t c = w->capOut; growHost( w->hOut, w->capOut, outN ); growDev( w->dOut, outN, c ); }
    { size_t c = w->capCls; growHost( w->hCls, w->capCls, clsN ); growDev( w->dCls, clsN, c ); }
    { size_t c = w->capMap; growHost( w->hMap, w->capMap, mapN ); growDev( w->dMap, mapN, c ); }

    // ---- host pack: pageable picture buffer -> pinned staging
    for( int y = 0; y < p.bandH + 2 * kHalo; y++ )
    {
      memcpy( w->hIn + ( size_t ) y * inPitch, p.src + ( ptrdiff_t ) ( y - kHalo ) * p.srcStride - kHalo, inPitch * sizeof( Pel ) );
    }
    memcpy( w->hMap, p.ctuSetIdx, mapN * sizeof( int16_t ) );
    if( p.numAps > 0 )
    {
      memcpy( w->hAps, p.apsCoef, p.numAps * kSetStride * sizeof( int16_t ) );
      memcpy( w->hAps + kMaxAps * kSetStride, p.apsClip, p.numAps * kSetStride * sizeof( int16_t ) );
    }
    if( mode == MODE_FILTER ) memcpy( w->hCls, p.cpuClass, clsN );
    const auto t1 = clk::now();

    KParams k;
    k.picWidth = p.picWidth;  k.bandY0 = p.bandY0;  k.bandH = p.bandH;  k.ctuSize = p.ctuSize;
    k.numCtuCols = p.numCtuCols;  k.bitDepth = p.bitDepth;  k.vbH = p.vbH;  k.vbPos = p.vbPos;
    k.inPitch = inPitch;  k.clsStride = clsStride;  k.outPitch = p.picWidth;

    uint64_t h2d = inN * sizeof( Pel ) + mapN * sizeof( int16_t );
    cudaStream_t s = w->stream;

    // ---- H2D
    ALF_CUDA_CHECK( cudaEventRecord( w->ev[0], s ) );
    ALF_CUDA_CHECK( cudaMemcpyAsync( w->dIn,  w->hIn,  inN  * sizeof( Pel ),     cudaMemcpyHostToDevice, s ) );
    ALF_CUDA_CHECK( cudaMemcpyAsync( w->dMap, w->hMap, mapN * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
    if( p.numAps > 0 )
    {
      const size_t n = ( size_t ) p.numAps * kSetStride;
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->dCoef + kNumFixedSets * kSetStride, w->hAps, n * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->dClip + kNumFixedSets * kSetStride, w->hAps + kMaxAps * kSetStride, n * sizeof( int16_t ), cudaMemcpyHostToDevice, s ) );
      h2d += 2 * n * sizeof( int16_t );
    }
    if( mode == MODE_FILTER )
    {
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->dCls, w->hCls, clsN, cudaMemcpyHostToDevice, s ) );
      h2d += clsN;
    }
    ALF_CUDA_CHECK( cudaEventRecord( w->ev[1], s ) );

    // ---- kernels
    if( mode != MODE_FILTER ) launchAlfClassify( w->dIn, w->dMap, w->dCls, k, s );
    ALF_CUDA_CHECK( cudaGetLastError() );
    ALF_CUDA_CHECK( cudaEventRecord( w->ev[2], s ) );
    launchAlfFilterLuma( w->dIn, w->dMap, w->dCls, w->dCoef, w->dClip, w->dOut, k, s );
    ALF_CUDA_CHECK( cudaGetLastError() );
    ALF_CUDA_CHECK( cudaEventRecord( w->ev[3], s ) );

    // ---- D2H
    ALF_CUDA_CHECK( cudaMemcpyAsync( w->hOut, w->dOut, outN * sizeof( Pel ), cudaMemcpyDeviceToHost, s ) );
    uint64_t d2h = outN * sizeof( Pel );
    if( p.outClass && mode != MODE_FILTER )
    {
      ALF_CUDA_CHECK( cudaMemcpyAsync( w->hCls, w->dCls, clsN, cudaMemcpyDeviceToHost, s ) );
      d2h += clsN;
    }
    ALF_CUDA_CHECK( cudaEventRecord( w->ev[4], s ) );
    ALF_CUDA_CHECK( cudaEventSynchronize( w->ev[4] ) );
    const auto t2 = clk::now();

    // ---- host unpack: only the CTUs handled by the GPU
    for( int r = 0; r < ctuRows; r++ )
    {
      const int y0 = r * p.ctuSize, h = std::min( p.ctuSize, p.bandH - y0 );
      for( int c = 0; c < p.numCtuCols; c++ )
      {
        if( p.ctuSetIdx[r * p.numCtuCols + c] < 0 ) continue;
        const int x0 = c * p.ctuSize, wd = std::min( p.ctuSize, p.picWidth - x0 );
        for( int y = y0; y < y0 + h; y++ )
        {
          memcpy( p.dst + ( ptrdiff_t ) y * p.dstStride + x0, w->hOut + ( size_t ) y * p.picWidth + x0, wd * sizeof( Pel ) );
        }
      }
    }
    if( p.outClass ) memcpy( p.outClass, w->hCls, clsN );
    const auto t3 = clk::now();

    float e01 = 0, e12 = 0, e23 = 0, e34 = 0;
    cudaEventElapsedTime( &e01, w->ev[0], w->ev[1] );
    cudaEventElapsedTime( &e12, w->ev[1], w->ev[2] );
    cudaEventElapsedTime( &e23, w->ev[2], w->ev[3] );
    cudaEventElapsedTime( &e34, w->ev[3], w->ev[4] );

    t.hostPackMs   += ms( t0, t1 );
    t.h2dMs        += e01;
    t.classifyMs   += e12;
    t.filterMs     += e23;
    t.d2hMs        += e34;
    t.hostUnpackMs += ms( t2, t3 );
    t.totalMs      += ms( t0, t3 );
    t.bytesH2D     += h2d;
    t.bytesD2H     += d2h;
  }
};

Backend* createCudaBackend()
{
  return new CudaBackend;   // init() reports failure when no device is available
}

} // namespace alfgpu
