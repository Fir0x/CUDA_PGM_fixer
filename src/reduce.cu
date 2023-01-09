#include "fix_gpu.hh"

namespace CustomCore
{
#ifdef GPU_FIX
    __inline__ __device__ int warp_reduce(int val)
    {
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(~0, val, offset);
        return val;
    }

    __global__ void reduce_kernel(int *data, int size, int *res)
    {
        uint tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid + 32 < size)
        {
            int sum = 0;

            sum = warp_reduce(data[tid]);

            if (tid % 32 == 0)
                atomicAdd(&res[0], sum);
        }
        else if (tid < size) 
        {
             atomicAdd(&res[0], data[tid]);
        }
    }

    __global__ void reduce_kernel0(int *data, int size, int *res)
    {
        // Really naive reduce, need improvements
        uint id = blockDim.x * blockIdx.x + threadIdx.x;
        if (id < size)
            atomicAdd(&res[0], data[id]);
    }

    int reduce(Image &to_fix)
    {
        int *total;
        cudaMalloc_custom(&total, sizeof(int));
        cudaMemset(total, 0, sizeof(int));

        int size = to_fix.height * to_fix.width;
        int nbBlocks = std::ceil((float)size / NB_THREADS);

        reduce_kernel0<<<nbBlocks, NB_THREADS>>>(to_fix.gpu_values, size, total);
        int res = 0;        
        cudaMemcpy(&res, total, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(total);
        //printf("Total %d\n", res);
        return res;
    }
#endif
}