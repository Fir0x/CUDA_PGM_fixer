#include "fix_gpu.hh"

#include <iostream>

namespace Core
{
    typedef thrust::device_vector<int>::iterator intiter;
    typedef thrust::counting_iterator<int>     countiter;

    struct mapping_functor
    {
        __host__ __device__
        int operator()(const thrust::tuple<int, int>& tuple)
        {
            int x = tuple.get<0>();
            int i = tuple.get<1>();

            if (i % 4 == 0)
                return x + 1;
            else if (i % 4 == 1)
                return x - 5;
            else if (i % 4 == 2)
                return x + 3;
            else if (i % 4 == 3)
                return x - 8;
        }
    };

    // Apply map to fix pixels
    void step_2(thrust::device_vector<int>& to_fix, size_t image_size)
    {
        std::cout << "Step 2 ref" << std::endl;

        std::cout << "Accumulate before: " << thrust::reduce(to_fix.begin(), to_fix.end(), 0) << std::endl;

        thrust::counting_iterator<int> idxfirst(0);
        thrust::counting_iterator<int> idxlast = idxfirst + image_size;

        auto first = thrust::make_zip_iterator(thrust::make_tuple(to_fix.begin(), idxfirst));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(to_fix.begin() + image_size, idxlast));

        thrust::transform(first, last, to_fix.begin(), mapping_functor());

        std::cout << "Accumulate after: " << thrust::reduce(to_fix.begin(), to_fix.end(), 0) << std::endl;
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

