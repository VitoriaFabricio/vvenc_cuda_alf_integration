# Standalone tools for the bit-exact ALF GPU work
#
#   make bench            -> alf_bench   (needs nvcc + NVML, i.e. a CUDA toolkit with the driver installed)
#   make test_core        -> test_alf_core (needs libvvenc built in ../../build/<BUILD>, CPU only)
#
# Variables: CUDA_ARCH (default native), NVCC, BUILD (cmake build dir name, default base)

NVCC      ?= nvcc
CUDA_ARCH ?= native
BUILD     ?= base
ROOT      := ../..

bench: alf_bench
alf_bench: alf_bench.cu $(ROOT)/source/Lib/CommonLib/AlfGpuHost.cpp $(ROOT)/source/Lib/CommonLib/AlfGpuCore.h $(ROOT)/source/Lib/CommonLib/AlfGpu.h $(ROOT)/source/Lib/CommonLib/cuda/AlfGpuKernels.cuh
	$(NVCC) -O3 -std=c++14 -arch=$(CUDA_ARCH) -lineinfo -o $@ alf_bench.cu $(ROOT)/source/Lib/CommonLib/AlfGpuHost.cpp -lnvidia-ml

test_core: test_alf_core
test_alf_core: test_alf_core.cpp $(ROOT)/source/Lib/CommonLib/AlfGpuCore.h
	g++ -O2 -std=gnu++14 -flto=auto -DTARGET_SIMD_X86=1 -DHAVE_PTHREADS \
	    -I$(ROOT)/include -I$(ROOT)/build/$(BUILD) -I$(ROOT)/source/Lib -I$(ROOT)/source/Lib/CommonLib -I$(ROOT)/source/Lib/CommonLib/x86 \
	    test_alf_core.cpp $(ROOT)/lib/release-static/libvvenc.a -lpthread -o $@

clean:
	rm -f alf_bench test_alf_core

.PHONY: bench test_core clean
