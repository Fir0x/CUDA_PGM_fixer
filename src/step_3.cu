#include "fix_gpu.hh"

#include <iostream>

namespace Core
{
    // Histogram equalization
    void step_3(thrust::device_vector<int> to_fix)
    {
        std::cout << "Step 3 ref" << std::endl;
        // 1. Histogram
        // TODO
        // 2. Compute the inclusive sum scan of the histogram
        // TODO
        // 3. Find the first non-zero value in the cumulative histogram
        // TODO
        // 4. Apply the map transformation of the histogram equalization
        // TODO
    }
} // namespace Core

namespace CustomCore {

    void step_3(int* to_fix, ImageInfo imageInfo)
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