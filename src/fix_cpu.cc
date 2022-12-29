#include "fix_cpu.hh"
#include "image.hh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

void fix_image_cpu(Image& to_fix)
{
    const int image_size = to_fix.width * to_fix.height;

    // #1 Compact

    // Build predicate vector

    std::vector<int> predicate(to_fix.buffer.size(), 0);

    constexpr int garbage_val = -27;
    for (std::size_t i = 0; i < to_fix.buffer.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            predicate[i] = 1;

    // Compute the exclusive sum of the predicate

    std::exclusive_scan(predicate.begin(), predicate.end(), predicate.begin(), 0);

    // Scatter to the corresponding addresses
    auto it = std::find(to_fix.buffer.begin(), to_fix.buffer.end(), -27);
    std::cout << "It info before: S " << it - to_fix.buffer.begin() << " E " << to_fix.buffer.end() - it << std::endl;

    for (std::size_t i = 0; i < predicate.size(); ++i)
        if (to_fix.buffer[i] != garbage_val)
            to_fix.buffer[predicate[i]] = to_fix.buffer[i];

    it = std::find(to_fix.buffer.begin(), to_fix.buffer.end(), -27);
    std::cout << "It info after: S " << it - to_fix.buffer.begin() << " E " << to_fix.buffer.end() - it << std::endl;

    // #2 Apply map to fix pixels

    std::cout << "Accumulate before: " << std::accumulate(to_fix.buffer.begin(), to_fix.buffer.end(), 0) << std::endl;
    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            to_fix.buffer[i] += 1;
        else if (i % 4 == 1)
            to_fix.buffer[i] -= 5;
        else if (i % 4 == 2)
            to_fix.buffer[i] += 3;
        else if (i % 4 == 3)
            to_fix.buffer[i] -= 8;
    }
    std::cout << "Accumulate after: " << std::accumulate(to_fix.buffer.begin(), to_fix.buffer.end(), 0) << std::endl;

    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[to_fix.buffer[i]];

    std::cout << "Histogram accumulation: " << std::accumulate(histo.begin(), histo.end(), 0) << std::endl;

    // Compute the inclusive sum scan of the histogram

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });

    const int cdf_min = *first_none_zero;
    std::cout << "First cdf_min: " << cdf_min << std::endl;

    // Apply the map transformation of the histogram equalization

    std::transform(to_fix.buffer.data(), to_fix.buffer.data() + image_size, to_fix.buffer.data(),
        [image_size, cdf_min, &histo](int pixel)
            {
                return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
            }
    );

    std::cout << "Last accumulation: " << std::accumulate(to_fix.buffer.begin(), to_fix.buffer.end(), 0) << std::endl;
}