#include "fix_gpu.hh"

#include <iostream>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>

namespace Core
{
    // Compact
    void step_1(thrust::device_vector<int> to_fix)
    {
        std::cout << "Step 1 ref" << std::endl;

        // 1 Build the predicate vector
        // std::vector<int> predicate(to_fix, 0);
        // thrust::device_vector<int> predicate((imageInfo.width + imageInfo.pitch) * height);
        // struct generate_mask
        // {
        //     __host__ __device__ bool operator()(const int x)
        //     {
        //         return x != -27;
        //     }
        // };

        // 2 Exclusive sum of the predicate
        // thrust::exclusive_scan(thrust::host, predicate.begin(), predicate.end(), predicate.begin(), 0);

        // 3 Scatter to the corresponding addresses
        // TODO
        // thrust::scatter
        thrust::copy_if(to_fix.begin(), to_fix.end(), to_fix.begin(), [=] __device__ __host__ (const int x) { return x != -27;});
    }

} // namespace Core

namespace CustomCore
{
    // Compact
    void step_1([[maybe_unused]] int *to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        constexpr int garbage_val = -27;

        std::cout << "Step 1 custom" << std::endl;
        // 1 Build the predicate vector
        // TODO

        // 2 Exclusive sum of the predicate
        // TODO

        // 3 Scatter to the corresponding addresses
        // TODO
    }
} // namespace CustomCore
