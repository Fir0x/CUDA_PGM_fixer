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

    __global__ void reduce_kernel0(int *data, int size, int *res)
    {
        uint tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < size)
        {
            int sum = 0;

            sum = warp_reduce(data[tid]);

            if (tid % 32 == 0)
                atomicAdd(&res[0], sum);
        }
    }

    __global__ void reduce_kernel1(int *data, int size, int *res)
    {
        uint tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid < size)
        {
            int sum = 0;
            for (int i = tid; i < size; i+= gridDim.x * blockDim.x) {
                sum += data[i];
            }

            sum = warp_reduce(sum);

            if (tid % 32 == 0)
                atomicAdd(&res[0], sum);
        }
    }

    __global__ void reduce_kernel2(int *data, int size, int *res)
    {
        uint tid = (blockDim.x * blockIdx.x + threadIdx.x) * 4 ;
        if (tid < size)
        {
            int sum = 0;
            for (int i = tid; i < size / 4; i+= gridDim.x * blockDim.x) {
                sum += data[i] + data[i + 1] + data[i + 2] + data[i + 3];
                //int4 val = reinterpret_cast<int4*>(data)[i];
                //sum += val.x + val.y + val.z + val.w;
            }

            sum = warp_reduce(sum);

            if (tid % 32 == 0)
                atomicAdd(&res[0], sum);
        }
    }

    int reduce(Image &to_fix)
    {
        int *total;
        cudaMalloc_custom(&total, sizeof(int), __LINE__, __FILE__);
        cudaMemset(total, 0, sizeof(int));

        int size = to_fix.height * to_fix.width;
        int nbBlocks = std::ceil((float)size / NB_THREADS);

        reduce_kernel1<<<nbBlocks / 4, NB_THREADS>>>(to_fix.gpu_values, size, total);
        int res = 0;        
        cudaMemcpy(&res, total, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(total);
        //printf("Total %d\n", res);
        return res;
    }
#endif
}