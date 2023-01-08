#include "fix_gpu.hh"
#include <cstring>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

void Core::fix_image_gpu(Image &to_fix)
{
    // Send image to GPU
    thrust::device_vector<int> d_fix(to_fix.buffer);
    const size_t image_size = to_fix.width * to_fix.height;

    Core::step_1(d_fix);
    Core::step_2(d_fix, image_size);
    Core::step_3(d_fix, image_size);

    // Get data back to CPU
    thrust::copy(d_fix.begin(), d_fix.end(), to_fix.buffer.begin());
}

void CustomCore::fix_image_gpu_custom(Image &to_fix)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Send image to GPU
    const int image_size = to_fix.width * to_fix.height;
    int *pinedBuffer;
    cudaError_t err;
    err = cudaMallocHost((void**)&pinedBuffer, sizeof(int) * to_fix.buffer.size());
    if (err)
    {
        std::cout << "Host Alloc ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
    std::memcpy(pinedBuffer, to_fix.buffer.data(), to_fix.buffer.size() * sizeof(int));

    int *image_data;
    size_t length_pitch;

    // cudaError_t err = cudaMallocPitch(&image_data, &length_pitch, sizeof(int) * to_fix.width, to_fix.height);
    cudaMallocAsync(&image_data, sizeof(int) * to_fix.buffer.size(), stream);
    // cudaMemcpy2D(image_data, length_pitch, to_fix.buffer.data(), 0, to_fix.width * sizeof(int), to_fix.height, cudaMemcpyHostToDevice);
    err = cudaMemcpyAsync(image_data, pinedBuffer, sizeof(int) * to_fix.buffer.size(), cudaMemcpyHostToDevice, stream);
    if (err)
    {
        std::cout << "Memcpy ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(err);
    }

    CustomCore::ImageInfo imageInfo = {to_fix.width, to_fix.height, length_pitch, to_fix.buffer.size()};

    CustomCore::step_1(image_data, imageInfo, stream);
    CustomCore::step_2(image_data, imageInfo, stream);
    CustomCore::step_3(image_data, imageInfo, stream);

    std::cout << "End of steps " << std::endl;

    // Get data back to CPU
    // cudaMemcpy2D(to_fix.buffer.data(), 0, image_data, length_pitch, to_fix.width * sizeof(int), to_fix.height, cudaMemcpyDeviceToHost);
    err = cudaMemcpyAsync(to_fix.buffer.data(), image_data, sizeof(int) * to_fix.buffer.size(), cudaMemcpyDeviceToHost, stream);
    if (err != 0) {
        std::cout << cudaGetErrorString(err) << std::endl;
        exit(err);
    }
    //std::memcpy(to_fix.buffer.data(), pinedBuffer, to_fix.buffer.size() * sizeof(int));

    cudaFreeHost(pinedBuffer);

    cudaStreamDestroy(stream);
}