#include "fix_gpu.hh"

#include <iostream>
#include <thrust/binary_search.h>
#include <thrust/find.h>

namespace Core
{

    struct non_zero_functor
    {
        __device__ __host__ bool operator()(const int x)
        {
            return x != 0;
        }
    };

    struct apply_functor
    {
        thrust::device_ptr<int> histogram;
        const int cdf_min;
        const size_t image_size;

        apply_functor(thrust::device_ptr<int> _histogram, int _cdf_min, size_t _image_size)
            : histogram(_histogram), cdf_min(_cdf_min), image_size(_image_size)
        {
        }

        __device__ __host__ float operator()(const int pixel)
        {
            return roundf(((histogram[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
        }
    };

    // Histogram equalization
    void step_3(thrust::device_vector<int> &to_fix, size_t image_size)
    {
        // 1. Histogram
        thrust::device_vector<int> histogram(256, 0);
        thrust::device_vector<int> to_fix_tmp(to_fix.begin(), to_fix.begin() + image_size);
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
        apply_functor apply_instance(histogram.data(), cdf_min, image_size);

        thrust::transform(to_fix.begin(), to_fix.begin() + image_size, to_fix.begin(), apply_instance);
    }
} // namespace Core

namespace CustomCore
{
    __global__ void build_histogram(int *to_fix, int *histo, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
        {
            int index_histo = to_fix[id];
            // TODO Easy to improve
            atomicAdd(histo + index_histo, 1);
        }
    }

    __global__ void find_first_non_zero(int *histo, int work_per_thread, int *find_first_non_zero)
    {
        __shared__ int min_non_zeros[16];
        
        int start_id = threadIdx.x * work_per_thread;
        int min = histo[start_id]; 

        // Will iterate on a part of the array
        // Separe te work on x thread group
        for (int i = start_id + 1; i < start_id + work_per_thread; i++)
        {
            int val = histo[i];
            if (min == 0 || (val != 0 && val < min))
            {
                min = val;
            }
        }
        min_non_zeros[threadIdx.x] = min;

        __syncthreads();
        
        min = 0;
        // A thread alone is here to get the final result
        if (threadIdx.x == 0)
        {
            for (int i = 0; i < NB_THREADS / work_per_thread; i++)
            {
                int val = min_non_zeros[i];
                if (min == 0 || (val != 0 && val < min))
                {
                    min = val;
                }
            }
            *find_first_non_zero = min;
        }
    }

    __global__ void histo_equalization(int *to_fix, int *histo, int *first_non_zero, int size)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id < size)
        {
            int fix_val = to_fix[id];
            int histo_val = histo[fix_val];
            to_fix[id] = std::roundf(((histo_val - *first_non_zero) / static_cast<float>(size - *first_non_zero)) * 255.0f);
        }
    }
    void step_3([[maybe_unused]] int *to_fix, [[maybe_unused]] ImageInfo imageInfo)
    {
        //std::cout << "=== Start step 3 custom" << std::endl;
        int size = imageInfo.height * imageInfo.width;
        int nbBlocks = std::ceil((float)size / NB_THREADS);

        // 1. Histogram
        int *histogram;
        cudaMalloc_custom(&histogram, sizeof(int) * 256);
        cudaMemset(histogram, 0, sizeof(int) * 256);
        build_histogram<<<nbBlocks, NB_THREADS>>>(to_fix, histogram, size);
        checkKernelError("build_histogram");
        //cudaDeviceSynchronize();

        // { // debug
        //     thrust::device_ptr<int> tmp_histogram = thrust::device_pointer_cast(histogram);
        //     std::cout << "Histogram accumulation: " << thrust::reduce(tmp_histogram, tmp_histogram + 256, 0) << std::endl;
        // }

        // 2. Compute the inclusive sum scan of the histogram
        scan(histogram, 256, true);

        // 3. Find the first non-zero value in the cumulative histogram
        int *first_non_zero;
        cudaMalloc_custom(&first_non_zero, sizeof(int));
        int work_per_thread = 16;
        find_first_non_zero<<<1, 16>>>(histogram, work_per_thread, first_non_zero);
        checkKernelError("find_first_non_zero");
        //cudaDeviceSynchronize();

        // { // debug
        //     thrust::device_ptr<int> tmp_first = thrust::device_pointer_cast(first_non_zero);
        //     std::cout << "First cdf_min: " << *tmp_first << std::endl;
        // }

        // 4. Apply the map transformation of the histogram equalization
        histo_equalization<<<nbBlocks, NB_THREADS>>>(to_fix, histogram, first_non_zero, size);
        checkKernelError("histo_equalization");
        //cudaDeviceSynchronize();

        // { // debug
        //     thrust::device_ptr<int> tmp_fix = thrust::device_pointer_cast(to_fix);
        //     std::cout << "Last accumulation: " << thrust::reduce(tmp_fix, tmp_fix + size, 0) << std::endl;
        // }

        cudaFree(histogram);
        cudaFree(first_non_zero);
    }
} // namespace CustomCore