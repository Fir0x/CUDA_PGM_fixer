#include "fix_gpu.hh"
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#ifdef REF_GPU_FIX
void fix_image_gpu(Image &to_fix)
{
    // Send image to GPU
    thrust::device_vector<int> d_fix(to_fix.buffer);
    const size_t image_size = to_fix.width * to_fix.height;

    Core::step_1(d_fix);
    Core::step_2(d_fix, image_size);
    Core::step_3(d_fix, image_size);
    to_fix.gpu_values = d_fix;
}

void Core::fix_image_gpu_copy(Image &to_fix)
{
    // Get data back to CPU
    thrust::device_vector<int> values = to_fix.gpu_values;
    thrust::copy(values.begin(), values.end(), to_fix.buffer.begin());
}

#elif defined GPU_FIX
void fix_image_gpu_custom(Image &to_fix)
{
    // Send image to GPU
    int *image_data;

    CustomCore::cudaMalloc_custom(&image_data, sizeof(int) * to_fix.buffer.size(), __LINE__, __FILE__);
    cudaError_t err = cudaMemcpy(image_data, to_fix.buffer.data(), sizeof(int) * to_fix.buffer.size(), cudaMemcpyHostToDevice);
    if (err != 0)
        exit(err);

    CustomCore::ImageInfo imageInfo = {to_fix.width, to_fix.height, to_fix.buffer.size()};

    CustomCore::step_1(image_data, imageInfo);
    CustomCore::step_2(image_data, imageInfo);
    CustomCore::step_3(image_data, imageInfo);
    
    to_fix.gpu_values = image_data;
}

// Get data back to CPU
void CustomCore::fix_image_gpu_custom_copy(Image &to_fix)
{
    int *values = to_fix.gpu_values;
    const int image_size = to_fix.width * to_fix.height;
    cudaError_t err = cudaMemcpy(&to_fix.buffer[0], values, sizeof(int) * image_size, cudaMemcpyDeviceToHost);
    if (err != 0)
    {
        std::cout << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
    cudaFree(values);
}

#endif