/* -----------------------------------------------------------------------------
AlfGpuHost.cpp  -  host reference loops + host-emulation backend

The emulation backend mimics the GPU data flow (pack band with halo -> classify ->
filter -> unpack) but runs the shared core on the CPU. It exists to prove that
the integration (indices, halos, VB, borders, per-CTU sets) is bit-exact with
VVenC on machines without a GPU, and as a CPU baseline for the benchmark.
------------------------------------------------------------------------------- */

#include "AlfGpu.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mutex>
#include <memory>

namespace alfgpu
{

static inline int ctuOf( const BandParams& p, int x, int yInBand )
{
  return ( yInBand / p.ctuSize ) * p.numCtuCols + x / p.ctuSize;
}

void hostClassifyBand( const Pel* in, ptrdiff_t inStride, const BandParams& p, uint8_t* cls, int clsStride )
{
  const int shift = p.bitDepth + 4;
  for( int yb = 0; yb < p.bandH; yb += 4 )
  {
    for( int x = 0; x < p.picWidth; x += 4 )
    {
      if( p.ctuSetIdx[ctuOf( p, x, yb )] < 0 ) continue;
      int c, t;
      classifyBlock4x4( in + yb * inStride + x, inStride, p.bandY0 + yb, shift, p.vbH, p.vbPos, c, t );
      cls[( yb >> 2 ) * clsStride + ( x >> 2 )] = ( uint8_t ) ( c | ( t << 5 ) );
    }
  }
}

void hostFilterBand( const Pel* in, ptrdiff_t inStride, const BandParams& p, const int16_t* coefTab, const int16_t* clipTab,
                     const uint8_t* cls, int clsStride, Pel* out, ptrdiff_t outStride )
{
  const int maxVal = ( 1 << p.bitDepth ) - 1;
  for( int yb = 0; yb < p.bandH; yb++ )
  {
    for( int x = 0; x < p.picWidth; x++ )
    {
      const int set = p.ctuSetIdx[ctuOf( p, x, yb )];
      if( set < 0 ) continue;
      const uint8_t c   = cls[( yb >> 2 ) * clsStride + ( x >> 2 )];
      const int classIdx = c & 31, tr = c >> 5;
      const int16_t* cf = coefTab + set * kSetStride + classIdx * kNumCoeff;
      const int16_t* cl = clipTab + set * kSetStride + classIdx * kNumCoeff;
      out[yb * outStride + x] = filterLumaSample( in + yb * inStride + x, inStride, p.bandY0 + yb, cf, cl, tr, p.vbH, p.vbPos, maxVal );
    }
  }
}

// copies the rectangles of CTUs with setIdx >= 0 from (src,srcStride) to (dst,dstStride); both point to (0,bandY0)
static void copyCtuAreas( const BandParams& p, const Pel* src, ptrdiff_t srcStride, Pel* dst, ptrdiff_t dstStride )
{
  const int rows = ( p.bandH + p.ctuSize - 1 ) / p.ctuSize;
  for( int r = 0; r < rows; r++ )
  {
    const int y0 = r * p.ctuSize, h = std::min( p.ctuSize, p.bandH - y0 );
    for( int c = 0; c < p.numCtuCols; c++ )
    {
      if( p.ctuSetIdx[r * p.numCtuCols + c] < 0 ) continue;
      const int x0 = c * p.ctuSize, w = std::min( p.ctuSize, p.picWidth - x0 );
      for( int y = y0; y < y0 + h; y++ )
      {
        memcpy( dst + y * dstStride + x0, src + y * srcStride + x0, w * sizeof( Pel ) );
      }
    }
  }
}

class HostEmuBackend : public Backend
{
  struct Worker
  {
    std::vector<Pel>     band;   // input band with halo ("device" input)
    std::vector<Pel>     out;    // filtered band ("device" output)
    std::vector<uint8_t> cls;
  };
  std::vector<int16_t> m_coef, m_clip;   // kMaxSets * kSetStride
  std::vector<std::unique_ptr<Worker>> m_workers;
  std::mutex m_mutex;

public:
  const char* name() const override { return "host-emu"; }

  bool init( const int16_t* fixedCoef, const int16_t* fixedClip, int, int, int numWorkers ) override
  {
    m_coef.assign( kMaxSets * kSetStride, 0 );
    m_clip.assign( kMaxSets * kSetStride, 0 );
    memcpy( m_coef.data(), fixedCoef, kNumFixedSets * kSetStride * sizeof( int16_t ) );
    memcpy( m_clip.data(), fixedClip, kNumFixedSets * kSetStride * sizeof( int16_t ) );
    m_workers.resize( std::max( 1, numWorkers ) );
    return true;
  }

  void filterLumaBand( const BandParams& p, int mode, int worker, Timing& t ) override
  {
    using clk = std::chrono::steady_clock;
    auto ms   = []( clk::time_point a, clk::time_point b ) { return std::chrono::duration<double, std::milli>( b - a ).count(); };
    const auto t0 = clk::now();

    Worker* w;
    {
      std::lock_guard<std::mutex> lock( m_mutex );
      if( worker >= ( int ) m_workers.size() ) m_workers.resize( worker + 1 );
      if( !m_workers[worker] ) m_workers[worker].reset( new Worker );
      w = m_workers[worker].get();
    }

    const int inW = p.picWidth + 2 * kHalo, inH = p.bandH + 2 * kHalo;
    const int clsStride = p.picWidth >> 2;
    w->band.resize( ( size_t ) inW * inH );
    w->out.resize( ( size_t ) p.picWidth * p.bandH );
    w->cls.resize( ( size_t ) clsStride * ( ( p.bandH + 3 ) >> 2 ) );

    // per-call coefficient table: fixed sets + APS sets of this picture (copy per call -> thread safe)
    std::vector<int16_t> coef( m_coef ), clip( m_clip );
    if( p.numAps > 0 )
    {
      memcpy( &coef[kNumFixedSets * kSetStride], p.apsCoef, p.numAps * kSetStride * sizeof( int16_t ) );
      memcpy( &clip[kNumFixedSets * kSetStride], p.apsClip, p.numAps * kSetStride * sizeof( int16_t ) );
    }

    // "H2D": pack band with halo
    const auto t1 = clk::now();
    for( int y = 0; y < inH; y++ )
    {
      memcpy( &w->band[( size_t ) y * inW], p.src + ( ptrdiff_t ) ( y - kHalo ) * p.srcStride - kHalo, inW * sizeof( Pel ) );
    }
    if( mode == MODE_FILTER ) memcpy( w->cls.data(), p.cpuClass, w->cls.size() );
    const auto t2 = clk::now();

    const Pel* in = &w->band[( size_t ) kHalo * inW + kHalo];
    if( mode != MODE_FILTER ) hostClassifyBand( in, inW, p, w->cls.data(), clsStride );
    const auto t3 = clk::now();
    hostFilterBand( in, inW, p, coef.data(), clip.data(), w->cls.data(), clsStride, w->out.data(), p.picWidth );
    const auto t4 = clk::now();

    // "D2H" + unpack
    copyCtuAreas( p, w->out.data(), p.picWidth, p.dst, p.dstStride );
    if( p.outClass ) memcpy( p.outClass, w->cls.data(), w->cls.size() );
    const auto t5 = clk::now();

    t.hostPackMs   += 0;
    t.h2dMs        += ms( t1, t2 );
    t.classifyMs   += ms( t2, t3 );
    t.filterMs     += ms( t3, t4 );
    t.hostUnpackMs += ms( t4, t5 );
    t.totalMs      += ms( t0, t5 );
    t.bytesH2D     += ( uint64_t ) inW * inH * sizeof( Pel );
    t.bytesD2H     += ( uint64_t ) p.picWidth * p.bandH * sizeof( Pel );
  }
};

#if VVENC_CUDA_ALF
Backend* createCudaBackend();   // AlfGpuCuda.cu, returns nullptr if no usable device
#endif

Backend* createBackend( const int16_t* fixedCoef, const int16_t* fixedClip, int maxPicWidth, int maxBandH, int numWorkers )
{
  const char* emu = getenv( "VVENC_ALF_GPU_EMU" );
#if VVENC_CUDA_ALF
  if( !( emu && atoi( emu ) ) )
  {
    Backend* b = createCudaBackend();
    if( b && b->init( fixedCoef, fixedClip, maxPicWidth, maxBandH, numWorkers ) ) return b;
    delete b;
    fprintf( stderr, "[AlfGpu] no usable CUDA device, falling back to host emulation\n" );
  }
#else
  ( void ) emu;
#endif
  Backend* b = new HostEmuBackend;
  b->init( fixedCoef, fixedClip, maxPicWidth, maxBandH, numWorkers );
  return b;
}

} // namespace alfgpu
