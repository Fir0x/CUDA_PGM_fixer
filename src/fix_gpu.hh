#pragma once

#include "image.hh"
#include <thrust/device_vector.h>

void fix_image_gpu(Image& to_fix);
void fix_image_gpu_custom(Image& to_fix);

namespace Core
{
    void step_1(thrust::device_vector<int>& to_fix);
    void step_2(thrust::device_vector<int>& to_fix);
    void step_3(thrust::device_vector<int>& to_fix);
} // namespace Core

namespace CustomCore
{
    struct ImageInfo
    {
        int width;
        int height;
        size_t device_pitch;
    };

    void step_1(int* to_fix, ImageInfo imageInfo);
    void step_2(int* to_fix, ImageInfo imageInfo);
    void step_3(int* to_fix, ImageInfo imageInfo);
} // namespace CustomCore
