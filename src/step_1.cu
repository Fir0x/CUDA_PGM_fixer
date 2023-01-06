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
    __global__ void build_predicate(int *to_fix, int *predicate, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
            predicate[id] = to_fix[id] != -27;
    }

    __global__ void scatter(int *to_fix, int *to_fix_cpy, int *predicate, int size)
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
    void step_1([[maybe_unused]] int *to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        //std::cout << "=== Start step 1 custom" << std::endl;

        int size = imageInfo.corrupted_size;
        int nbBlocks = std::ceil((float)size / NB_THREADS);

        // TODO /!\ for first version, don't use streams, will add it later

        // 1 Build the predicate vector
        int *predicate;
        cudaMalloc_custom(&predicate, sizeof(int) * size);
        std::cout << "Start predicate kernel" << std::endl;
        build_predicate<<<nbBlocks, NB_THREADS>>>(to_fix, predicate, size);
        checkKernelError("build_predicate");
        //cudaDeviceSynchronize();

        // thrust::device_ptr<int> pred_copy_tmp;
        // { // debug
        //     thrust::device_ptr<int> pred_tmp = thrust::device_pointer_cast(predicate);
        //     int *pred_copy;
        //     cudaMalloc_custom(&pred_copy, sizeof(int) * size);
        //     pred_copy_tmp = thrust::device_pointer_cast(pred_copy);
        //     thrust::copy(pred_tmp, pred_tmp + size, pred_copy_tmp);
        //     // check if scan worked
        //     thrust::exclusive_scan(pred_copy_tmp, pred_copy_tmp + size, pred_copy_tmp, 0);
        // }

        // 2 Exclusive sum of the predicate
        std::cout << "Start scan" << std::endl;
        scan(predicate, size, false);

        // { // debug
        //     thrust::device_ptr<int> pred_tmp = thrust::device_pointer_cast(predicate);
        //     for (int i = 0; i < size; i++)
        //     {
        //         if (pred_tmp[i] != pred_copy_tmp[i])
        //         {
        //             std::cout << "ERROR in scan at: " << i << " Ref: " << pred_copy_tmp[i] << " Get: " << pred_tmp[i] << std::endl;
        //             break;
        //         }
        //     }
        // }

        // 3 Scatter to the corresponding addresses
        const int image_size = imageInfo.width * imageInfo.height;
        // { // debug
        //     thrust::device_ptr<int> tmp_fix = thrust::device_pointer_cast(to_fix);
        //     std::cout << "BS Fix ";
        //     for (int i = 0; i < 20; i++)
        //     {
        //         std::cout << tmp_fix[i] << " ; ";
        //     }
        //     std::cout << std::endl;
        //     auto it = thrust::find(tmp_fix, tmp_fix + size, -27);
        //     std::cout << "It info before: S " << it - tmp_fix << " E " << tmp_fix + size - it << std::endl;
        // }

        int *to_fix_cpy;
        cudaMalloc_custom(&to_fix_cpy, sizeof(int) * size);
        cudaMemcpy(to_fix_cpy, to_fix, sizeof(int) * size, cudaMemcpyDeviceToDevice);
        //std::cout << "Start scatter" << std::endl;
        scatter<<<nbBlocks, NB_THREADS>>>(to_fix, to_fix_cpy, predicate, size);
        checkKernelError("scatter");
        //cudaDeviceSynchronize();

        // { // debug
        //     thrust::device_ptr<int> tmp_fix = thrust::device_pointer_cast(to_fix);
        //     std::cout << "AS Fix ";
        //     for (int i = 0; i < 20; i++)
        //     {
        //         std::cout << tmp_fix[i] << " ; ";
        //     }
        //     std::cout << std::endl;
        //     auto it = std::find(tmp_fix, tmp_fix + size, -27);
        //     std::cout << "It info after: S " << it - tmp_fix << " E " << tmp_fix + size - it << std::endl;
        // }

        cudaFree(predicate);
        cudaFree(to_fix_cpy);
    }
} // namespace CustomCore
