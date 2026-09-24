#pragma once
#include <cuda_runtime.h>

#ifdef __CUDACC__
    #define __HOST__   __host__
    #define __DEVICE__ __device__
    #define __GLOBAL__ __global__
#else
    #define __HOST__
    #define __DEVICE__
    #define __GLOBAL__
#endif

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// allocate gpu space
template <typename T>
void allocateGpuBuffer(T*& d_ptr, size_t count) { cudaMalloc(&d_ptr, count * sizeof(T)); }

template <typename T>
void releaseGpuBuffer(T*& d_ptr) { if (d_ptr) { cudaFree(d_ptr); d_ptr = nullptr; } }


