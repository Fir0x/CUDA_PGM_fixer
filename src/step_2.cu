#include "fix_gpu.hh"

#include <iostream>

namespace Core
{
    void step_2([[maybe_unused]] Image& to_fix)
    {
#ifdef REF_GPU_FIX
        std::cout << "Step 2 ref" << std::endl;
#else
        std::cout << "Step 2 custom" << std::endl;
#endif
    }
} // namespace Core
