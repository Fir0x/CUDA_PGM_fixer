#include "fix_gpu.hh"

#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

namespace Core
{
    struct generate_mask
    {
        __host__ __device__ bool operator()(const int x)
        {
            return x != -27;
        }
    };

    // Compact
    void step_1(thrust::device_vector<int>& to_fix)
    {
        thrust::copy_if(to_fix.begin(), to_fix.end(), to_fix.begin(), generate_mask());
    }

} // namespace Core

namespace CustomCore
{
    __global__ void build_predicate(int* to_fix, int* predicate, int size)  
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
            predicate[id] = to_fix[id] != -27;
    }

    __global__ void scatter(int* to_fix, int* to_fix_cpy, int* predicate, int size) 
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x; 
        if (id < size) {
            int new_index = predicate[id];
            to_fix[new_index] = to_fix_cpy[id];
        }
    }

    // Compact
    void step_1([[maybe_unused]] int *to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        std::cout << "Step 1 custom" << std::endl;

        int size = imageInfo.corrupted_size;
        int nbBlocks = std::ceil((float)size / NB_THREADS);

        // TODO /!\ for first version, don't use streams, will add it later

        // 1 Build the predicate vector
        int* predicate;
        cudaMalloc_custom(&predicate, sizeof(int) * size);
        std::cout << "Start predicate kernel" << std::endl;
        build_predicate<<<nbBlocks, NB_THREADS>>>(to_fix, predicate, size);
        checkKernelError("build_predicate");
        cudaDeviceSynchronize();

        // 2 Exclusive sum of the predicate
        std::cout << "Start scan" << std::endl;
        scan(predicate, size, false);
        
        // 3 Scatter to the corresponding addresses
        int *to_fix_cpy;
        cudaMalloc_custom(&to_fix_cpy, sizeof(int) * size);
        cudaMemcpy(to_fix_cpy, to_fix, sizeof(int) * size, cudaMemcpyDeviceToDevice);
        std::cout << "Start scatter" << std::endl;
        scatter<<<nbBlocks, NB_THREADS>>>(to_fix, to_fix_cpy, predicate, size);
        checkKernelError("scatter");
        cudaDeviceSynchronize();

        /*int *tmp;
        cudaMalloc(&tmp, sizeof(int) * size);
        cudaMemcpy(tmp, to_fix, sizeof(int) * size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < imageInfo.height * imageInfo.width; i++) {
            std::cout << "Go" << std::endl;
            int val = tmp[i];
            std::cout << "Geted values" << std::endl;
            if (val == -27) 
            {
                std::cout << "Bad values left" << std::endl;
            }
        }
        std::cout << "Whoop whoop" << std::endl;*/

        cudaFree(predicate);
        cudaFree(to_fix_cpy);
    }
} // namespace CustomCore
