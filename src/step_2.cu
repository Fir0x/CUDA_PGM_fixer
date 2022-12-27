#include "fix_gpu.hh"

#include <iostream>

namespace Core
{
    typedef thrust::device_vector<int>::iterator intiter;
    typedef thrust::counting_iterator<int> countiter;

    struct mapping_functor
    {
        __host__ __device__ int operator()(const thrust::tuple<int, int> &tuple)
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
    void step_2(thrust::device_vector<int> &to_fix, size_t image_size)
    {
        thrust::counting_iterator<int> idxfirst(0);
        thrust::counting_iterator<int> idxlast = idxfirst + image_size;

        auto first = thrust::make_zip_iterator(thrust::make_tuple(to_fix.begin(), idxfirst));
        auto last = thrust::make_zip_iterator(thrust::make_tuple(to_fix.begin() + image_size, idxlast));

        thrust::transform(first, last, to_fix.begin(), mapping_functor());
    }
} // namespace Core

namespace CustomCore
{
    __global__ void map_fix(int *to_fix, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
        {
            if (id % 4 == 0)
                to_fix[id] += 1;
            else if (id % 4 == 1)
                to_fix[id] -= 5;
            else if (id % 4 == 2)
                to_fix[id] += 3;
            else if (id % 4 == 3)
                to_fix[id] -= 8;
        }
    }
    // Apply map to fix pixels
    void step_2([[maybe_unused]] int *to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        std::cout << "Step 2 custom" << std::endl;
        int size = imageInfo.height * imageInfo.width;
        int nbBlocks = std::ceil((float)size / NB_THREADS);
        map_fix<<<nbBlocks, NB_THREADS>>>(to_fix, size);
    }
} // namespace CustomCore
