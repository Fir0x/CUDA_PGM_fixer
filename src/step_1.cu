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

    __global__ void scatter(int* to_fix, int* predicate, int size) 
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x; 
        if (id < size)
            to_fix[predicate[id]] = to_fix[id];
    }

    // Compact
    void step_1([[maybe_unused]] int *to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        std::cout << "Step 1 custom" << std::endl;

        constexpr int garbage_val = -27;
        int size = imageInfo.height * imageInfo.width;
        int nbBlocks = std::ceil((float)size / NB_THREADS);

        // TODO /!\ for first version, don't use streams, will add it later

        // 1 Build the predicate vector
        int* predicate;
        cudaMalloc(&predicate, sizeof(int) * size);
        build_predicate<<<nbBlocks, NB_THREADS>>>(to_fix, predicate, size);

        // 2 Exclusive sum of the predicate
        scan(predicate, size, false);

        // 3 Scatter to the corresponding addresses
        scatter<<<nbBlocks, NB_THREADS>>>(to_fix, predicate, size);

        cudaFree(predicate);
    }
} // namespace CustomCore
