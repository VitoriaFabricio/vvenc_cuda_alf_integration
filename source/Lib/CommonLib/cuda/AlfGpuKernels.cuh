/* -----------------------------------------------------------------------------
AlfGpuKernels.cuh - CUDA kernels for bit-exact ALF luma (classification + filter)

Both kernels stage a tile of the input band in shared memory and then call the
very same functions the CPU reference uses (AlfGpuCore.h), so the arithmetic is
identical by construction. Shared by the VVenC backend (AlfGpuCuda.cu) and the
standalone benchmark (test/alfgpu/alf_bench.cu).

Input band layout (device): (picWidth + 2*kHalo) x (bandH + 2*kHalo) samples,
row pitch inPitch; d_in points to the top-left of the *halo*.
------------------------------------------------------------------------------- */
#pragma once

#include "../AlfGpuCore.h"
#include <cuda_runtime.h>

namespace alfgpu
{

struct KParams
{
  int picWidth, bandY0, bandH, ctuSize, numCtuCols, bitDepth, vbH, vbPos;
  int inPitch;     // samples
  int clsStride;   // bytes per row of 4x4 blocks (= picWidth / 4)
  int outPitch;    // samples
};

// ---- classification: one thread per 4x4 block, a block of threads covers 128 x 32 samples
static constexpr int CLS_TX = 32, CLS_TY = 8;
static constexpr int CLS_TW = CLS_TX * 4 + 6;   // tile width  (3 halo samples each side)
static constexpr int CLS_TH = CLS_TY * 4 + 6;   // tile height

__global__ void kAlfClassify( const Pel* __restrict__ d_in, const int16_t* __restrict__ d_ctuSet,
                              uint8_t* __restrict__ d_cls, KParams k )
{
  __shared__ Pel tile[CLS_TH][CLS_TW];

  const int x0 = blockIdx.x * CLS_TX * 4;   // band coordinates of the tile's first sample
  const int y0 = blockIdx.y * CLS_TY * 4;
  const int tid = threadIdx.y * CLS_TX + threadIdx.x;

  for( int i = tid; i < CLS_TW * CLS_TH; i += CLS_TX * CLS_TY )
  {
    const int r = i / CLS_TW, c = i - r * CLS_TW;
    const int gx = x0 - 3 + c, gy = y0 - 3 + r;   // band coords, valid range [-kHalo, W + kHalo)
    Pel v = 0;
    if( gx >= -kHalo && gx < k.picWidth + kHalo && gy >= -kHalo && gy < k.bandH + kHalo )
    {
      v = d_in[( size_t ) ( gy + kHalo ) * k.inPitch + ( gx + kHalo )];
    }
    tile[r][c] = v;
  }
  __syncthreads();

  const int bx = x0 + threadIdx.x * 4, by = y0 + threadIdx.y * 4;
  if( bx >= k.picWidth || by >= k.bandH ) return;
  if( d_ctuSet[( by / k.ctuSize ) * k.numCtuCols + bx / k.ctuSize] < 0 ) return;

  int c, t;
  classifyBlock4x4( &tile[threadIdx.y * 4 + 3][threadIdx.x * 4 + 3], CLS_TW, k.bandY0 + by, k.bitDepth + 4, k.vbH, k.vbPos, c, t );
  d_cls[( by >> 2 ) * k.clsStride + ( bx >> 2 )] = ( uint8_t ) ( c | ( t << 5 ) );
}

// ---- filter: 32 x 8 threads, each thread filters 4 samples of a column -> 32 x 32 samples per block
static constexpr int FLT_TX = 32, FLT_TY = 8, FLT_ROWS = 4;
static constexpr int FLT_BW = FLT_TX, FLT_BH = FLT_TY * FLT_ROWS;
static constexpr int FLT_TW = FLT_BW + 6, FLT_TH = FLT_BH + 6;

__global__ void kAlfFilterLuma( const Pel* __restrict__ d_in, const int16_t* __restrict__ d_ctuSet,
                                const uint8_t* __restrict__ d_cls,
                                const int16_t* __restrict__ d_coef, const int16_t* __restrict__ d_clip,
                                Pel* __restrict__ d_out, KParams k )
{
  __shared__ Pel tile[FLT_TH][FLT_TW];

  const int x0 = blockIdx.x * FLT_BW, y0 = blockIdx.y * FLT_BH;
  const int tid = threadIdx.y * FLT_TX + threadIdx.x;

  for( int i = tid; i < FLT_TW * FLT_TH; i += FLT_TX * FLT_TY )
  {
    const int r = i / FLT_TW, c = i - r * FLT_TW;
    const int gx = x0 - 3 + c, gy = y0 - 3 + r;
    Pel v = 0;
    if( gx >= -kHalo && gx < k.picWidth + kHalo && gy >= -kHalo && gy < k.bandH + kHalo )
    {
      v = d_in[( size_t ) ( gy + kHalo ) * k.inPitch + ( gx + kHalo )];
    }
    tile[r][c] = v;
  }
  __syncthreads();

  const int x = x0 + threadIdx.x;
  if( x >= k.picWidth ) return;
  const int maxVal = ( 1 << k.bitDepth ) - 1;

#pragma unroll
  for( int j = 0; j < FLT_ROWS; j++ )
  {
    const int ly = threadIdx.y + j * FLT_TY;   // row inside the block
    const int y  = y0 + ly;                    // band row
    if( y >= k.bandH ) break;
    const int set = d_ctuSet[( y / k.ctuSize ) * k.numCtuCols + x / k.ctuSize];
    if( set < 0 ) continue;
    const uint8_t cl   = d_cls[( y >> 2 ) * k.clsStride + ( x >> 2 )];
    const int classIdx = cl & 31, tr = cl >> 5;
    const int off      = set * kSetStride + classIdx * kNumCoeff;
    d_out[( size_t ) y * k.outPitch + x] =
      filterLumaSample( &tile[ly + 3][threadIdx.x + 3], FLT_TW, k.bandY0 + y, d_coef + off, d_clip + off, tr, k.vbH, k.vbPos, maxVal );
  }
}

inline void launchAlfClassify( const Pel* d_in, const int16_t* d_ctuSet, uint8_t* d_cls, const KParams& k, cudaStream_t s )
{
  dim3 block( CLS_TX, CLS_TY );
  dim3 grid( ( k.picWidth + CLS_TX * 4 - 1 ) / ( CLS_TX * 4 ), ( k.bandH + CLS_TY * 4 - 1 ) / ( CLS_TY * 4 ) );
  kAlfClassify<<<grid, block, 0, s>>>( d_in, d_ctuSet, d_cls, k );
}

inline void launchAlfFilterLuma( const Pel* d_in, const int16_t* d_ctuSet, const uint8_t* d_cls, const int16_t* d_coef,
                                 const int16_t* d_clip, Pel* d_out, const KParams& k, cudaStream_t s )
{
  dim3 block( FLT_TX, FLT_TY );
  dim3 grid( ( k.picWidth + FLT_BW - 1 ) / FLT_BW, ( k.bandH + FLT_BH - 1 ) / FLT_BH );
  kAlfFilterLuma<<<grid, block, 0, s>>>( d_in, d_ctuSet, d_cls, d_coef, d_clip, d_out, k );
}

} // namespace alfgpu
