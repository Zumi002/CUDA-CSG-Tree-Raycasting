#pragma once

#include <cuda_runtime.h>
#include "../Utils/CSGUtils.cuh"
#include "../Utils/Float3Utils.cuh"
#include "RayHit.cuh"

struct Ray
{
    float3 origin;
    float3 direction;

    inline __device__ Ray(float3 o, float3 d) : origin(o), direction(d) {
        // Normalize direction
        float invLen = 1.0f / sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
        direction.x *= invLen;
        direction.y *= invLen;
        direction.z *= invLen;
    }
    inline __device__ float3 computePosition(const float& t) const {
        return origin + t * direction;
    }
};

struct RayHitMinimal
{
    float t;
    unsigned char hit;
    unsigned char primitiveType;
    short primitiveIdx;

    inline __device__ RayHitMinimal(float t, unsigned char hit, short primitiveIdx) :
        t(t), hit(hit), primitiveIdx(primitiveIdx){}
    inline __device__ RayHitMinimal()
    {
        t = -1;
        hit = CSG::CSGRayHit::Miss;
        primitiveIdx = -1;
    }
        
};
