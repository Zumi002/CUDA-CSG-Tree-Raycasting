#pragma once

#include <cuda_runtime.h>

struct RayHit {
    bool hit;
    float t;
    float3 position;
    float3 normal;
    int primitiveIdx;
};