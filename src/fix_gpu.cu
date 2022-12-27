#include "fix_gpu.hh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

void fix_image_gpu(Image& to_fix)
{
    // Send image to GPU 
    thrust::device_vector<int> d_fix (to_fix.buffer);
    
    Core::step_1(d_fix);
    Core::step_2(d_fix);
    Core::step_3(d_fix);

    // Get data back to CPU
    thrust::copy(to_fix.buffer.begin(), to_fix.buffer.end(), d_fix.begin());
}

void fix_image_gpu_custom(Image& to_fix)
{
    // Send image to GPU 
    const int image_size = to_fix.width * to_fix.height;
    int* image_data; 
    size_t length_pitch; 

    //cudaError_t err = cudaMallocPitch(&image_data, &length_pitch, sizeof(int) * to_fix.width, to_fix.height);
    cudaError_t err = cudaMalloc(&image_data, sizeof(int) * to_fix.width * to_fix.height);
    if (err != 0)
        exit(err);
    //cudaMemcpy2D(image_data, length_pitch, to_fix.buffer.data(), 0, to_fix.width * sizeof(int), to_fix.height, cudaMemcpyHostToDevice);
    cudaMemcpy(image_data, to_fix.buffer.data(), to_fix.buffer.size(), cudaMemcpyHostToDevice);

    CustomCore::ImageInfo imageInfo = { to_fix.width, to_fix.height, length_pitch };

    CustomCore::step_1(image_data, imageInfo);
    CustomCore::step_2(image_data, imageInfo);
    CustomCore::step_3(image_data, imageInfo);

    // Get data back to CPU
    //cudaMemcpy2D(to_fix.buffer.data(), 0, image_data, length_pitch, to_fix.width * sizeof(int), to_fix.height, cudaMemcpyDeviceToHost);
    cudaMemcpy(to_fix.buffer.data(), image_data, to_fix.buffer.size(), cudaMemcpyDeviceToHost);
}