#include "fix_gpu.hh"

#include <iostream>
#include <thrust/binary_search.h>
#include <thrust/find.h>

namespace Core
{

    struct non_zero_functor
    {
        __device__ __host__
        bool operator()(const int x)
        {
            return x != 0;
        }
    };

    struct apply_functor
    {
        thrust::device_vector<int>& histogram;
        const int cdf_min;
        const size_t image_size;

        apply_functor(thrust::device_vector<int>& _histogram, int _cdf_min, size_t _image_size)
            : histogram(_histogram), cdf_min(_cdf_min), image_size(_image_size)
        {
        }

        __device__ __host__
        float operator()(const int pixel)
        {
            return pixel + 1;//roundf(((histogram[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
        }
    };

    // Histogram equalization
    void step_3([[maybe_unused]] thrust::device_vector<int>& to_fix)
    {
        std::cout << "Step 3 ref" << std::endl;
        // 1. Histogram
        thrust::device_vector<int> histogram(256, 0);
        thrust::device_vector<int> to_fix_tmp(to_fix);
        thrust::sort(to_fix_tmp.begin(), to_fix_tmp.end());

        thrust::counting_iterator<int> search_begin(0);
        thrust::upper_bound(to_fix_tmp.begin(), to_fix_tmp.end(),
                            search_begin, search_begin + 256,
                            histogram.begin());

        thrust::adjacent_difference(histogram.begin(), histogram.end(),
                                    histogram.begin());

        // 2. Compute the inclusive sum scan of the histogram
        thrust::inclusive_scan(thrust::device, histogram.begin(), histogram.end(), histogram.begin());

        // 3. Find the first non-zero value in the cumulative histogram
        auto first_none_zero = thrust::find_if(thrust::device, histogram.begin(), histogram.end(), non_zero_functor());
        const int cdf_min = *first_none_zero;

        // 4. Apply the map transformation of the histogram equalization
        size_t image_size = to_fix.size();
        apply_functor apply_instance(histogram, cdf_min, image_size);

        thrust::transform(to_fix.begin(), to_fix.end(), to_fix.begin(), apply_instance);
    }
} // namespace Core

namespace CustomCore
{

    void step_3([[maybe_unused]] int *to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        std::cout << "Step 3 custom" << std::endl;
        // 1. Histogram
        // TODO
        // 2. Compute the inclusive sum scan of the histogram
        // TODO
        // 3. Find the first non-zero value in the cumulative histogram
        // TODO
        // 4. Apply the map transformation of the histogram equalization
        // TODO
    }
} // namespace CustomCore