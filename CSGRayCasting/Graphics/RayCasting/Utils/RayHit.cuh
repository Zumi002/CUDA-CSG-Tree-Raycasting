#pragma once

#include <cuda_runtime.h>

struct __align__(32) RayHit {
    float3 position;
    float3 normal;
    int primitiveIdx;
    bool hit;
};