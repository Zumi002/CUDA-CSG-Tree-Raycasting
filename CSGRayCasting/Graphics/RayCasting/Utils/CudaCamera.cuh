#pragma once

#include <cuda_runtime.h>

class CudaCamera
{
    public:
    float3 position;
    float3 forward;
    float3 right;
    float3 up;
    float fov;
};