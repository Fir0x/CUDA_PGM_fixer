#pragma once

#include "image.hh"
#include <thrust/device_vector.h>


void fix_image_gpu(Image& to_fix);
namespace Core
{
    void step_1(thrust::device_vector<int>& to_fix);
    void step_2(thrust::device_vector<int>& to_fix, size_t image_size);
    void step_3(thrust::device_vector<int>& to_fix, size_t image_size);
    void fix_image_gpu_copy(Image &to_fix);
} // namespace Core

void fix_image_gpu_custom(Image& to_fix);
namespace CustomCore
{
    #define NB_THREADS 256

    struct ImageInfo
    {
        int width;
        int height;
        size_t corrupted_size;
    };

    void fix_image_gpu_custom_copy(Image &to_fix);
    void step_1(int* to_fix, ImageInfo imageInfo);
    void step_2(int* to_fix, ImageInfo imageInfo);
    void step_3(int* to_fix, ImageInfo imageInfo);
    void scan(int *buffer, int size);
    void scan_inclusive(int* data);
    void cudaMalloc_custom(int** ptr, size_t size, int line, const char *file);
    void checkKernelError(std::string name);
    #ifdef GPU_FIX
    int reduce(Image &to_fix);
    #endif
} // namespace CustomCore
