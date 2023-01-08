#pragma once

#include "image.hh"
#include <thrust/device_vector.h>

namespace Core
{
    void fix_image_gpu(Image& to_fix);

    void step_1(thrust::device_vector<int>& to_fix);
    void step_2(thrust::device_vector<int>& to_fix, size_t image_size);
    void step_3(thrust::device_vector<int>& to_fix, size_t image_size);
} // namespace Core

namespace CustomCore
{
    #define NB_THREADS 256

    struct StreamPool
    {
        cudaStream_t imageAllocStream;
        cudaStream_t kernelStream;
        cudaStream_t device2HostStream;
        cudaStream_t host2DeviceStream;
    };

    struct ImageInfo
    {
        int width;
        int height;
        size_t device_pitch;
        size_t corrupted_size;
    };

    void step_1(int* to_fix, ImageInfo imageInfo, cudaStream_t streamPool);
    void step_2(int* to_fix, ImageInfo imageInfo, cudaStream_t streamPool);
    void step_3(int* to_fix, ImageInfo imageInfo, cudaStream_t streamPool);
    void scan(int *buffer, int size, bool inclusive, cudaStream_t);

    void cudaMalloc_custom(int** ptr, size_t size);
    void cudaMallocAsync_custom(int** ptr, size_t size, cudaStream_t stream);
    void checkKernelError(std::string name);

    void fix_image_gpu_custom(Image& to_fix, const StreamPool& streamPool);
} // namespace CustomCore
