#pragma once

#include "Ray.cuh"
#include "../CSGTree/CSGTree.cuh"


//
// All with help of amazing source for sdf: https://iquilezles.org/articles/distfunctions/
//

//
// ---- headers ----
//

__device__ inline float SphereSD(
    const float3& p,
    const CudaPrimitivePos& spherePosition,
    const Parameters& sphereParameters
);

__device__ inline float CubeSD(
    const float3& p,
    const CudaPrimitivePos& cubePosition,
    const Parameters& cubeParameters
);

__device__ inline float CylinderSD(
    const float3& p,
    const CudaPrimitivePos& cylinderPosition,
    const Parameters& cylinderParameters
);

//
// ---- code ----
//

__device__ float SphereSD(
    const float3& p,
    const CudaPrimitivePos& spherePosition,
    const Parameters& sphereParameters
)
{
    return length(p - make_float3(spherePosition.x, spherePosition.y, spherePosition.z)) -
        sphereParameters.sphereParameters.radius;
}

__device__ float CubeSD(
    const float3& p,
    const CudaPrimitivePos& cubePosition,
    const Parameters& cubeParameters
)
{
    float3 pc = p - make_float3(cubePosition.x, cubePosition.y, cubePosition.z);
    float3 q = abs(pc) - 
        0.5f* make_float3(cubeParameters.cubeParameters.size,
            cubeParameters.cubeParameters.size, 
            cubeParameters.cubeParameters.size);
    return length(max(q, make_float3(0.0, 0.0, 0.0))) +
            fmin(fmax(q.x, fmax(q.y, q.z)), 0.0f); 
}

__device__ float CylinderSD(
    const float3& p,
    const CudaPrimitivePos& cylinderPosition,
    const Parameters& cylinderParameters
)
{
    CylinderParameters params = cylinderParameters.cylinderParameters;
    float3 V = make_float3(params.axisX, params.axisY, params.axisZ);
    float3 ba = params.height * V;
    float3 a = make_float3(cylinderPosition.x, cylinderPosition.y, cylinderPosition.z) - 0.5f*ba;
    float3 pa = p - a;
   
    float baba = dot(ba, ba);
    float paba = dot(pa, ba);

    float x = length(baba * pa  - paba * ba) - params.radius * baba;
    float y = fabs(paba - baba * 0.5) - baba * 0.5;
    float x2 = x * x;
    float y2 = y * y * baba;
    float d = (fmax(x, y) < 0.0) ? -fmin(x2, y2) : (((x > 0.0) ? x2 : 0.0) + ((y > 0.0) ? y2 : 0.0));
    return ((d > 0) - (d < 0)) * sqrt(fabs(d))/baba;
}