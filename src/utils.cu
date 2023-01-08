#include "fix_gpu.hh"

namespace CustomCore
{
    void cudaMalloc_custom(int **ptr, size_t size)
    {
        cudaError_t err = cudaMalloc(ptr, size);
        if (err != 0)
        {
            std::cout << "Malloc ERROR: " << cudaGetErrorString(err) << std::endl;
            exit(err);
        }
    }

    void cudaMallocAsync_custom(int **ptr, size_t size, cudaStream_t stream)
    {
        cudaError_t err = cudaMallocAsync(ptr, size, stream);
        if (err != 0)
        {
            std::cout << "Malloc ERROR: " << cudaGetErrorString(err) << std::endl;
            exit(err);
        }
    }

    void checkKernelError(std::string name)
    {
        cudaError_t err = cudaGetLastError();
        if (err != 0)
        {
            std::cout << "Kernel ERROR: " << name << ": " << cudaGetErrorString(err) << std::endl;
            exit(err);
        }
    }
}