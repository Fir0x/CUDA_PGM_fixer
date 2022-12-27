#include "fix_gpu.hh"

#include <iostream>

namespace Core
{
    typedef thrust::device_vector<int>::iterator intiter;
    typedef thrust::counting_iterator<int>     countiter;

    // Apply map to fix pixels
    void step_2(thrust::device_vector<int> to_fix)
    {
        std::cout << "Step 2 ref" << std::endl;

        thrust::counting_iterator<int> idxfirst(0);
        thrust::counting_iterator<int> idxlast = idxfirst + to_fix.size();
        auto first = thrust::make_zip_iterator(thrust::make_tuple(to_fix.begin(), idxfirst));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(to_fix.end(), idxlast));
        thrust::transform(first, last, to_fix.begin(), to_fix.begin(), [=] __device__ __host__ (const thrust::tuple<intiter, countiter>& tuple)
        {
                int x = *tuple.get<0>();
                int i = *tuple.get<1>();

                if (i % 4 == 0)
                    return x + 1;
                else if (i % 4 == 1)
                    return x - 5;
                else if (i % 4 == 2)
                    return x + 3;
                else if (i % 4 == 3)
                    return x - 8;
                return x;
        });
    }
} // namespace Core

namespace CustomCore
{
    // Apply map to fix pixels
    void step_2([[maybe_unused]] int* to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        std::cout << "Step 2 custom" << std::endl;
    }
} // namespace CustomCore

