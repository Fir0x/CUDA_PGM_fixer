#include "fix_gpu.hh"

void fix_image_gpu(Image& to_fix)
{
    Core::step_1(to_fix);
    Core::step_2(to_fix);
    Core::step_3(to_fix);
}