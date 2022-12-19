#pragma once

#include "image.hh"

void fix_image_gpu(Image& to_fix);

namespace Core
{
    void step_1(Image& to_fix);
    void step_2(Image& to_fix);
    void step_3(Image& to_fix);
} // namespace CustomCore