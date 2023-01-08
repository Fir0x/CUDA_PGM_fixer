#include "fix_gpu.hh"

namespace CustomCore
{

    __inline__ __device__ void warp_reduce0(int *buffer_shared, int tid)
    {
        buffer_shared[tid] += buffer_shared[tid + 32];
        __syncwarp();
        buffer_shared[tid] += buffer_shared[tid + 16];
        __syncwarp();
        buffer_shared[tid] += buffer_shared[tid + 8];
        __syncwarp();
        buffer_shared[tid] += buffer_shared[tid + 4];
        __syncwarp();
        buffer_shared[tid] += buffer_shared[tid + 2];
        __syncwarp();
        buffer_shared[tid] += buffer_shared[tid + 1];
        __syncwarp();
    }

    __inline__ __device__ int warp_reduce1(int val)
    {
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(~0, val, offset);
        return val;
    }

    __device__ void reduce3(int *buffer_shared, int *blocks_sum, int block_id)
    {
        int sum = 0;
        uint tid = threadIdx.x;
        sum = warp_reduce1(buffer_shared[tid]);

        if (tid % 32 == 0)
            atomicAdd(&blocks_sum[block_id], sum);
    }

    __device__ void reduce2(int *buffer_shared, int *blocks_sum, int block_id)
    {
        int sum = 0;
        uint tid = threadIdx.x;

        __shared__ int sdata[NB_THREADS];
        sdata[tid] = buffer_shared[tid];
        __syncthreads();
        for (int s = blockDim.x / 2; s > 32; s >>= 1)
        {
            if (tid < s)
                sdata[tid] += sdata[tid + s];
            __syncthreads();
        }

        if (tid < 32)
        {
            warp_reduce0(sdata, tid);
            if (tid == 0)
                atomicAdd(&blocks_sum[block_id], sdata[0]);
        }
    }

    __device__ void reduce1(int *buffer_shared, int *blocks_sum, int block_id)
    {
        int sum = 0;
        uint tid = threadIdx.x;

        __shared__ int sdata[NB_THREADS];
        sdata[tid] = buffer_shared[tid];
        __syncthreads();

        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (tid < s)
                sdata[tid] += sdata[tid + s];

            __syncthreads();
        }

        if (tid == 0)
            atomicAdd(&blocks_sum[block_id], sdata[0]);
    }

    // Really simple reduce - lot of atomics
    __device__ void reduce0(int *buffer_shared, int *blocks_sum, int block_id)
    {
        // Really naive reduce, need improvements
        uint id = threadIdx.x;
        atomicAdd(&blocks_sum[block_id], buffer_shared[id]);
        __syncthreads();
    }

    __device__ void kogge_stone_scan0(int *buffer_shared, int local_index)
    {
        int plus = 1;
        int val;
        for (plus = 1; plus <= blockDim.x / 2; plus *= 2)
        {
            if (local_index + plus < blockDim.x)
            {
                val = buffer_shared[local_index];
            }
            __syncthreads();
            if (local_index + plus < blockDim.x)
            {
                if (local_index + plus != blockDim.x - 1)
                    buffer_shared[local_index + plus] += val;
                else
                    buffer_shared[local_index + plus] += val;
            }
            __syncthreads();
        }
    }

    __device__ void sklansky_scan0(int *buffer_shared, int local_index)
    {
        int val;
        int previous_pow = -1;
#pragma unroll
        for (int pow = 2; pow <= 256; pow <<= 1)
        {
            if ((local_index + previous_pow + 1) % pow == 0)
            {
                int val = buffer_shared[local_index];
#pragma unroll
                for (int i = 1; i <= pow >> 1; i++)
                {
                    buffer_shared[local_index + i] += val;
                }
            }
            __syncthreads();
            previous_pow = pow;
        }
    }

    // Do the decoupled loop back and add the sum on the buffer_shared[0]
    __device__ int decoupled_loop_back(int *buffer_shared, int *shared_state, int *blocks_sum, int block_manual_id)
    {
        int total_added = 0;
        int block_look_at = block_manual_id - 1;
        uint loop_back_in_progress = 1;
        while (block_look_at >= 0 && loop_back_in_progress)
        {
            int state = shared_state[block_look_at];

            // if (state == 0):
            //  do nothing, just wait block to be valid
            if (state == 1)
            {
                total_added += blocks_sum[block_look_at];
                block_look_at -= 1;
            }
            else if (state == 2)
            { // Pass on a P block -> the end of loop back
                total_added += blocks_sum[block_look_at];
                loop_back_in_progress = 0;
            }
        }

        buffer_shared[0] += total_added;
        return total_added;
    }

    /**
     * Kogge-Stone-Scan
     * Use neutral state to wait for total sum added
     **/
    template <typename T>
    __global__ void scan_kernel0(
        T *buffer, int size, int *shared_state, int *blocks_sum, int *block_order, bool inclusive)
    {
        __shared__ int block_manual_id;
        __shared__ int buffer_shared[NB_THREADS];

        // Block ordering
        if (threadIdx.x == 0)
            block_manual_id = atomicAdd(&block_order[0], 1);
        __threadfence_system();
        __syncthreads();

        int global_index = blockDim.x * block_manual_id + threadIdx.x;
        if (global_index < size)
        {
            int local_index = threadIdx.x;
            buffer_shared[local_index] = buffer[global_index];
            __syncthreads();

            // 1. Reduce in the block
            reduce3(buffer_shared, blocks_sum, block_manual_id);
            __syncthreads();
            __threadfence_system();

            if (threadIdx.x == 0)
                atomicAdd(&shared_state[block_manual_id], 1);
            __threadfence_system();

            // --- 2. decoupled loop back to get sum of other blocks
            int total_added = 0;
            if (threadIdx.x == 0)
                total_added = decoupled_loop_back(buffer_shared, shared_state, blocks_sum, block_manual_id);

            // Put on a state 3 to force other blocks to wait
            if (threadIdx.x == 0)
                atomicExch(&shared_state[block_manual_id], 3);
            __threadfence_system();

            if (threadIdx.x == 0)
                atomicAdd(&blocks_sum[block_manual_id], total_added);
            __threadfence_system();

            if (threadIdx.x == 0)
                atomicExch(&shared_state[block_manual_id], 2);
            __threadfence_system();
            // --- End of decoupled loop back

            // Everyone wait the thread 0 to finish decoupled loop back
            __syncthreads();

            // --- 3. Scan - Final scan on the block itself
            if (inclusive)
                sklansky_scan0(buffer_shared, local_index);
            else
            {
                int start_val = buffer_shared[threadIdx.x];
                __syncthreads();
                sklansky_scan0(buffer_shared, local_index);
                // kogge_stone_scan0(buffer_shared, local_index);
                if (threadIdx.x == 0)
                    buffer_shared[local_index] -= start_val - total_added;
                else
                    buffer_shared[local_index] -= start_val;
                __syncthreads();
            }

            __syncthreads();
            buffer[global_index] = buffer_shared[local_index];
            // --- End of Scan on the block
        }
    }

    void scan(int *buffer, int size, bool inclusive)
    {
        int nbBlocks = std::ceil((float)size / NB_THREADS);
        int *shared_state;
        cudaMalloc(&shared_state, sizeof(int) * nbBlocks);
        cudaMemset(shared_state, 0, sizeof(int) * nbBlocks);
        int *shared_sum;
        cudaMalloc(&shared_sum, sizeof(int) * nbBlocks);
        cudaMemset(shared_sum, 0, sizeof(int) * nbBlocks);
        int *block_order;
        cudaMalloc(&block_order, sizeof(int));
        cudaMemset(block_order, 0, sizeof(int));

        scan_kernel0<int><<<nbBlocks, NB_THREADS>>>(buffer,
                                                    size,
                                                    shared_state,
                                                    shared_sum,
                                                    block_order,
                                                    inclusive);

        cudaDeviceSynchronize();
        cudaFree(shared_sum);
        cudaFree(shared_state);
        cudaFree(block_order);
    }
}
