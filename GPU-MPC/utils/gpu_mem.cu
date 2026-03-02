// Author: Neha Jawalkar
// Copyright:
// 
// Copyright (c) 2024 Microsoft Research
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <chrono>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "helper_cuda.h"
#include "gpu_stats.h"
#include <cassert>

// #include <sys/types.h>

cudaMemPool_t mempool;

extern "C" void initGPUMemPool()
{
    int isMemPoolSupported = 0;
    int device = 0;
    // is it okay to use device=0?
    checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
                                           cudaDevAttrMemoryPoolsSupported, device));
    // printf("%d\n", isMemPoolSupported);
    assert(isMemPoolSupported);
    /* implicitly assumes that the device is 0 */

    checkCudaErrors(cudaDeviceGetDefaultMemPool(&mempool, device));
    uint64_t threshold = UINT64_MAX;
    checkCudaErrors(cudaMemPoolSetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold));

    uint64_t free_bytes = 0;
    uint64_t total_bytes = 0;
    checkCudaErrors(cudaMemGetInfo(&free_bytes, &total_bytes));

    uint64_t requested_bytes = 40ULL * (1ULL << 30);
    const char *warmup_gb_env = std::getenv("GPU_MEMPOOL_WARMUP_GB");
    if (warmup_gb_env != nullptr)
    {
        uint64_t warmup_gb = strtoull(warmup_gb_env, nullptr, 10);
        requested_bytes = warmup_gb * (1ULL << 30);
    }

    // Reserve at most 80% of currently free memory to avoid OOM.
    uint64_t bytes = std::min(requested_bytes, (free_bytes * 8) / 10);
    uint64_t *d_dummy_ptr = nullptr;
    while (bytes >= (1ULL << 28))
    {
        cudaError_t err = cudaMallocAsync(&d_dummy_ptr, bytes, 0);
        if (err == cudaSuccess)
        {
            checkCudaErrors(cudaFreeAsync(d_dummy_ptr, 0));
            break;
        }
        cudaGetLastError();
        bytes /= 2;
    }

    uint64_t reserved_read, threshold_read;
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReservedMemCurrent, &reserved_read));
    checkCudaErrors(cudaMemPoolGetAttribute(mempool, cudaMemPoolAttrReleaseThreshold, &threshold_read));
    printf("reserved memory: %lu %lu\n", reserved_read, threshold_read);
}

extern "C" uint8_t *gpuMalloc(size_t size_in_bytes)
{
    uint8_t *d_a;
    checkCudaErrors(cudaMallocAsync(&d_a, size_in_bytes, 0));
    return d_a;
}


extern "C" uint8_t *cpuMalloc(size_t size_in_bytes, bool pin)
{
    uint8_t *h_a;
    int err = posix_memalign((void **)&h_a, 32, size_in_bytes);
    assert(err == 0 && "posix memalign");
    if (pin)
        checkCudaErrors(cudaHostRegister(h_a, size_in_bytes, cudaHostRegisterDefault));
    return h_a;
}

extern "C" void gpuFree(void *d_a)
{
    checkCudaErrors(cudaFreeAsync(d_a, 0));
}

extern "C" void cpuFree(void *h_a, bool pinned)
{
    if (pinned)
        checkCudaErrors(cudaHostUnregister(h_a));
    free(h_a);
}

extern "C" uint8_t *moveToCPU(uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *h_a = cpuMalloc(size_in_bytes, true);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoGPUMem(uint8_t *d_a, uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveIntoCPUMem(uint8_t *h_a, uint8_t *d_a, size_t size_in_bytes, Stats *s)
{
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(h_a, d_a, size_in_bytes, cudaMemcpyDeviceToHost));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return h_a;
}

extern "C" uint8_t *moveToGPU(uint8_t *h_a, size_t size_in_bytes, Stats *s)
{
    uint8_t *d_a = gpuMalloc(size_in_bytes);
    auto start = std::chrono::high_resolution_clock::now();
    checkCudaErrors(cudaMemcpy(d_a, h_a, size_in_bytes, cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = end - start;
    if (s)
        s->transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
    return d_a;
}
