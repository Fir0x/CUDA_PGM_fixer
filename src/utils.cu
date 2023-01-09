#include "fix_gpu.hh"

namespace CustomCore
{
    void cudaMalloc_custom(int **ptr, size_t size, int line,const char* file)
    {
        cudaError_t err = cudaMalloc(ptr, size);
        if (err != 0)
        {
            std::cout << "Malloc ERROR: " << cudaGetErrorString(err) << " at " << file << ":" << line << std::endl;
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