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
    void step_1(thrust::device_vector<int> &to_fix)
    {
        thrust::copy_if(to_fix.begin(), to_fix.end(), to_fix.begin(), generate_mask());
    }

} // namespace Core

namespace CustomCore
{
    __global__ void build_predicate2(int *to_fix, int *predicate, int size)
    {
        // WIP TODO
        int id = (blockIdx.x * blockDim.x + threadIdx.x);

        // Grid stride loop pattern and vectorial access
        for (int i = id; i < size / 4; i += blockDim.x * gridDim.x)
        {
            int4 vals = reinterpret_cast<int4 *>(to_fix)[i];
            predicate[i] = vals.x != -27;
            predicate[i + 1] = vals.y != -27;
            predicate[i + 2] = vals.z != -27;
            predicate[i + 3] = vals.w != -27;
        }

        int remainder = size % 4;
        if (id == size % 4 && remainder != 0)
        {
            while (remainder)
            {
                int idx = size - remainder--;
                predicate[idx] = to_fix[idx] != -27;
            }
        }
    }
    __global__ void build_predicate1(int *to_fix, int *predicate, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        // Grid stride loop pattern
        for (int i = id; i < size; i += blockDim.x * gridDim.x)
            predicate[i] = to_fix[i] != -27;
    }

    __global__ void build_predicate0(int *to_fix, int *predicate, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
            predicate[id] = to_fix[id] != -27;
    }

    __global__ void scatter1(int *to_fix, int *predicate, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
        {
            int val = to_fix[id];
            __syncthreads();
            if (val != -27)
                to_fix[predicate[id]] = val;
        }
    }

    __global__ void scatter0(int *to_fix, int *to_fix_cpy, int *predicate, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
        {
            int val = to_fix_cpy[id];
            if (val != -27)
                to_fix[predicate[id]] = val;
        }
    }

    // Compact
    void step_1(int *to_fix, ImageInfo imageInfo)
    {
        int size = imageInfo.corrupted_size;
        int nbBlocks = std::ceil((float)size / NB_THREADS);

        // 1 Build the predicate vector
        int *predicate;
        cudaMalloc_custom(&predicate, sizeof(int) * size);
        build_predicate1<<<nbBlocks / 4, NB_THREADS>>>(to_fix, predicate, size);

        checkKernelError("build_predicate");

        // 2 Exclusive sum of the predicate
        std::cout << "Start scan" << std::endl;
        scan(predicate, size, false);

        // 3 Scatter to the corresponding addresses
        const int image_size = imageInfo.width * imageInfo.height;
        scatter1<<<nbBlocks, NB_THREADS>>>(to_fix, predicate, size);

        checkKernelError("scatter");

        cudaFree(predicate);
    }
} // namespace CustomCore
