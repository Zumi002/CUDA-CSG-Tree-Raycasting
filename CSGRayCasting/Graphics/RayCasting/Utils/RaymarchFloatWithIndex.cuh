#pragma once

#include <cuda_runtime.h>

struct FloatWithIndex
{
    float f;
    int idx;
    __device__ FloatWithIndex(float F = 0, int Idx = 0)
    {
        f = F;
        idx = Idx;
    }
};