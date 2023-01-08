#include "fix_gpu.hh"

namespace CustomCore
{
    // TODO Use best reduce possible
    __device__ void reduce_use_atomic(int *buffer_shared, int *blocks_sum, int block_id)
    {
        // Really naive reduce, need improvements
        uint id = threadIdx.x;
        atomicAdd(&blocks_sum[block_id], buffer_shared[id]);
    }

    __device__ void kogge_stone_scan_inclusive(int *buffer_shared, int local_index)
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

    __device__ void kogge_stone_scan_exclusive(int *buffer_shared, int local_index, int total_added)
    {
        int plus = 1;
        int val;

        // Todo find a way to remove this temp buffer
        __shared__ int buffer_exclu[NB_THREADS];
        buffer_exclu[threadIdx.x] = buffer_shared[threadIdx.x];
        __syncthreads();

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
        
        // TODO find a way to avoid this
        if (threadIdx.x == 0)
            buffer_shared[local_index] -= (buffer_exclu[local_index] - total_added);
        else
            buffer_shared[local_index] -= buffer_exclu[local_index];
        __syncthreads();
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
    __global__ void scan_kernel_1(
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
            reduce_use_atomic(buffer_shared, blocks_sum, block_manual_id);
            __syncthreads();

            __threadfence_system();

            if (threadIdx.x == 0)
            {
                atomicAdd(&shared_state[block_manual_id], 1);
            }

            // Be sure the atomic is fully executed
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

            // --- 3. Scan Kogge-Stone Way - Final scan on the block itself
            if (inclusive)
                kogge_stone_scan_inclusive(buffer_shared, local_index);
            else
                kogge_stone_scan_exclusive(buffer_shared, local_index, total_added);

            __syncthreads();
            buffer[global_index] = buffer_shared[local_index];
            // --- End of Scan on the block
        }
    }

    /**
     * Kogge-Stone-Scan
     * Use 2 arrays to store sum A and sum P
     **/
    template <typename T>
    __global__ void scan_kernel_2(T *buffer,
                                  int size,
                                  int *shared_state,
                                  int *blocks_sum_a,
                                  int *blocks_sum_p,
                                  int *block_order,
                                  bool inclusive)
    {
        if (threadIdx.x < size)
        {
            __shared__ int block_manual_id;
            __shared__ int buffer_shared[NB_THREADS];

            if (threadIdx.x == 0)
            {
                block_manual_id = atomicAdd(&block_order[0], 1);
            }

            __threadfence_system();
            __syncthreads();

            int local_index = threadIdx.x;
            int global_index = blockDim.x * block_manual_id + threadIdx.x;
            buffer_shared[local_index] = buffer[global_index];
            __syncthreads();

            // 1. Reduce in the block
            reduce_use_atomic(buffer_shared, blocks_sum_a, block_manual_id);
            __syncthreads();

            if (threadIdx.x == 0)
                atomicAdd(&shared_state[block_manual_id], 1);
            __threadfence_system();

            // --- 2. decoupled loop back to get sum of other blocks
            int total_added = 0;
            if (threadIdx.x == 0)
                total_added = decoupled_loop_back(buffer_shared, shared_state, blocks_sum_a, block_manual_id);

            if (threadIdx.x == 0)
                atomicExch(&blocks_sum_p[block_manual_id], total_added + blocks_sum_a[block_manual_id]);
            __threadfence_system();

            if (threadIdx.x == 0)
                atomicExch(&shared_state[block_manual_id], 2);
            __threadfence_system();
            // --- End of decoupled loop back

            // Everyone wait the thread 0 to finish decoupled loop back
            __syncthreads();

            // --- 3. Scan Kogge-Stone Way - Final scan on the block itself
            if (inclusive)
                kogge_stone_scan_inclusive(buffer_shared, local_index);
            else
                kogge_stone_scan_exclusive(buffer_shared, local_index, total_added);

            __syncthreads();
            buffer[global_index] = buffer_shared[local_index];
            // --- End of Scan on the block
        }
    }

    void scan(int *buffer, int size, bool inclusive, cudaStream_t stream)
    {
        int nbBlocks = std::ceil((float)size / NB_THREADS);
        int *shared_state;
        cudaMallocAsync_custom(&shared_state, sizeof(int) * nbBlocks, stream);
        cudaMemsetAsync(shared_state, 0, sizeof(int) * nbBlocks, stream);
        int *shared_sum;
        cudaMallocAsync_custom(&shared_sum, sizeof(int) * nbBlocks, stream);
        cudaMemsetAsync(shared_sum, 0, sizeof(int) * nbBlocks, stream);
        int *block_order;
        cudaMallocAsync_custom(&block_order, sizeof(int), stream);
        cudaMemsetAsync(block_order, 0, sizeof(int), stream);

        scan_kernel_1<int><<<nbBlocks, NB_THREADS, 0, stream>>>(buffer,
                                                     size,
                                                     shared_state,
                                                     shared_sum,
                                                     block_order,
                                                     inclusive);

        //cudaDeviceSynchronize();
        cudaFreeAsync(shared_sum, stream);
        cudaFreeAsync(shared_state, stream);
        cudaFreeAsync(block_order, stream);
    }
}
