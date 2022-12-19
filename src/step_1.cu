#include "fix_gpu.hh"

#include <iostream>

namespace Core
{
    void step_1([[maybe_unused]] Image& to_fix)
    {
#ifdef REF_GPU_FIX
        std::cout << "Step 1 ref" << std::endl;
#else
        std::cout << "Step 1 custom" << std::endl;
#endif
    }
} // namespace Core
