#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../Utils/PrimitiveRayIntersection.cuh"
#include "../Utils/CudaStack.cuh"
#include "../CSGTree/CSGTree.cuh"
#include "../Utils/CudaCamera.cuh"
#include "../Utils/PrimitiveSDF.cuh"
#include "../Utils/RaymarchFloatWithIndex.cuh"

#define MAXRAYMARCHINGSTACKSIZE 32
#define MAXMARCHES 60
#define RAYMARCHEPSILON 0.001f
#define RAYMARCH_H 0.001f
#define RAYMARCHFAR 1000.f

#define RAYMARCHSHAREDNODES 255

//
// ---- headers ----
//

__global__ void RaymarchingKernel(
    CudaCamera cam,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    RayHit* hits,
    int numberOfNodes,
    float width, float height
);

__global__ void RaymarchingKernelShared(
    CudaCamera cam,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    RayHit* hits,
    int numberOfNodes,
    float width, float height
);

__device__ inline FloatWithIndex MapCSGTreeWithIndex(
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    float3 ray,
    int numberOfNodes,
    CudaStack<FloatWithIndex, MAXRAYMARCHINGSTACKSIZE>& distanceStack
);

__device__ inline float MapCSGTree(
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    float3 ray,
    int numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack
);

__device__ inline float GetSD(
    CudaPrimitivePos primitivePos,
    Parameters primitiveParameters,
    int type,
    float3 ray
);

__device__ inline void UnionSDWithIndex(
    FloatWithIndex a,
    FloatWithIndex& b
);
__device__ inline void IntersectionSDWithIndex(
    FloatWithIndex a,
    FloatWithIndex& b
);
__device__ inline void DifferenceSDWithIndex(
    FloatWithIndex a,
    FloatWithIndex& b
);

__device__ inline void OperationWithIndex(
    int type,
    FloatWithIndex a,
    FloatWithIndex& b
);

__device__ inline void UnionSD(
    float a,
    float& b
);
__device__ inline void IntersectionSD(
    float a,
    float& b
);
__device__ inline void DifferenceSD(
    float a,
    float& b
);

__device__ inline void Operation(
    int type,
    float a,
    float& b
);

__device__ inline RayHit GetDetailedHitInfo(
    Ray ray,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    FloatWithIndex nodeIdx,
    int t,
    int numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack,
    float3 camPos,
    bool inside
);

__device__ inline float3 SDNormal(
    Ray ray,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    int numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack
);

//
// ---- code ----
//

__global__ void RaymarchingKernel(
    CudaCamera cam,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    RayHit* hits,
    int numberOfNodes,
    float width, float height
)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate normalized device coordinates (NDC)
    float u = ((float)x + 0.5f) / (width - 1);
    float v = ((float)y + 0.5f) / (height - 1);

    // Convert to screen space coordinates (-1 to 1)
    float nx = (2.0f * u - 1.0f) * (width / height) * tan(cam.fov / 2.0f);
    float ny = (1.0f - 2.0f * v) * tan(cam.fov / 2.0f);

    int pixelIdx = (y * (int)width + x);

    // Create ray from camera
    float3 rayOrigin = cam.position;
    float3 rayDirection = normalize(nx * cam.right + ny * cam.up + cam.forward);

    Ray ray(rayOrigin, rayDirection);
    float t = 0;
    bool inside = false;

    CudaStack<float, MAXRAYMARCHINGSTACKSIZE> distanceStack;

    int marches = 0;
    float mapResult = RAYMARCHFAR;

    // first unrolled to check if we are inside
    mapResult = MapCSGTree(nodes, primitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStack);
    ray.origin = ray.origin + fabs(mapResult) * ray.direction;
    t += fabs(mapResult);
    marches++;
    inside = mapResult < 0 ? true : false;
    while (marches < MAXMARCHES && fabs(mapResult) > RAYMARCHEPSILON && t < RAYMARCHFAR)
    {
        mapResult = MapCSGTree(nodes, primitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStack);
        ray.origin = ray.origin + fabs(mapResult) * ray.direction;
        t += fabs(mapResult);
        marches++;
    }

    CudaStack<FloatWithIndex, MAXRAYMARCHINGSTACKSIZE> distanceStackWithIndex;
    FloatWithIndex mapResultWithIdx = MapCSGTreeWithIndex(nodes, primitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStackWithIndex);

    hits[pixelIdx] = GetDetailedHitInfo(ray, nodes, primitivePos, primitiveParameters, mapResultWithIdx, t, numberOfNodes, distanceStack, cam.position, inside);

}

__global__ void RaymarchingKernelShared(
    CudaCamera cam,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    RayHit* hits,
    int numberOfNodes,
    float width, float height
)
{
    __shared__ CSGNode sharedNodes[RAYMARCHSHAREDNODES];
    __shared__ CudaPrimitivePos sharedPrimitivePos[RAYMARCHSHAREDNODES / 2 + 1];
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < numberOfNodes; i += blockDim.x * blockDim.y)
    {
        sharedNodes[i] = nodes[i];
    }
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < numberOfNodes / 2 + 1; i += blockDim.x * blockDim.y)
    {
        sharedPrimitivePos[i] = primitivePos[i];
    }
    __syncthreads();
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Calculate normalized device coordinates (NDC)
    float u = ((float)x + 0.5f) / (width - 1);
    float v = ((float)y + 0.5f) / (height - 1);

    // Convert to screen space coordinates (-1 to 1)
    float nx = (2.0f * u - 1.0f) * (width / height) * tan(cam.fov / 2.0f);
    float ny = (1.0f - 2.0f * v) * tan(cam.fov / 2.0f);

    int pixelIdx = (y * (int)width + x);

    // Create ray from camera
    float3 rayOrigin = cam.position;
    float3 rayDirection = normalize(nx * cam.right + ny * cam.up + cam.forward);

    Ray ray(rayOrigin, rayDirection);

    CudaStack<float, MAXRAYMARCHINGSTACKSIZE> distanceStack;

    int marches = 0;
    float mapResult = RAYMARCHFAR;
    float t = 0;
    int minDistPrimitiveIdx = -1;
    bool inside = false;

    // first unrolled to check if we are inside
    mapResult = MapCSGTree(sharedNodes, sharedPrimitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStack);
    ray.origin = ray.origin + fabs(mapResult) * ray.direction;
    t += fabs(mapResult);
    marches++;
    inside = mapResult < 0 ? true : false;
    while (marches < MAXMARCHES && fabs(mapResult) > RAYMARCHEPSILON && t < RAYMARCHFAR)
    {
        mapResult = MapCSGTree(sharedNodes, sharedPrimitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStack);
        ray.origin = ray.origin + fabs(mapResult) * ray.direction;
        t += fabs(mapResult);
        marches++;
    }
    CudaStack<FloatWithIndex, MAXRAYMARCHINGSTACKSIZE> distanceStackWithIndex;
    FloatWithIndex mapResultWithIdx = MapCSGTreeWithIndex(sharedNodes, sharedPrimitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStackWithIndex);

    hits[pixelIdx] = GetDetailedHitInfo(ray, sharedNodes, sharedPrimitivePos, primitiveParameters, mapResultWithIdx, t, numberOfNodes, distanceStack, cam.position, inside);
}

__device__ FloatWithIndex MapCSGTreeWithIndex(
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    float3 pos,
    int numberOfNodes,
    CudaStack<FloatWithIndex, MAXRAYMARCHINGSTACKSIZE>& distanceStack
)
{
    FloatWithIndex distWithIdx = FloatWithIndex(
        GetSD(
            primitivePos[nodes[0].primitiveIdx],
            primitiveParameters[nodes[0].primitiveIdx],
            nodes[0].type,
            pos),
        nodes[0].primitiveIdx
    );
    for (int i = 1; i < numberOfNodes; i++)
    {
        if (nodes[i].type<=CSGTree::NodeType::Intersection)
        {
            OperationWithIndex(
                nodes[i].type,
                distanceStack.pop(),
                distWithIdx
            );
        }
        else
        {
            distanceStack.push(distWithIdx);
            distWithIdx = FloatWithIndex(GetSD(
                primitivePos[nodes[i].primitiveIdx],
                primitiveParameters[nodes[i].primitiveIdx],
                nodes[i].type,
                pos),
                nodes[i].primitiveIdx
            );
        }
    }
    return distWithIdx;
}

__device__ float MapCSGTree(
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    float3 pos,
    int numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack
)
{
    float distWithIdx = GetSD(
        primitivePos[nodes[0].primitiveIdx],
        primitiveParameters[nodes[0].primitiveIdx],
        nodes[0].type,
        pos);
    for (int i = 1; i < numberOfNodes; i++)
    {
        if (nodes[i].type <= CSGTree::NodeType::Intersection)
        {
            Operation(
                nodes[i].type,
                distanceStack.pop(),
                distWithIdx
            );
        }
        else
        {
            distanceStack.push(distWithIdx);
            distWithIdx = GetSD(
                primitivePos[nodes[i].primitiveIdx],
                primitiveParameters[nodes[i].primitiveIdx],
                nodes[i].type,
                pos);
        }
    }
    return distWithIdx;
}

__device__ float GetSD(
    CudaPrimitivePos primitivePos,
    Parameters primitiveParameters,
    int type,
    float3 pos
)
{
    if (type == CSGTree::NodeType::Sphere)
    {
        return SphereSD(pos, primitivePos, primitiveParameters);
    }
    else if (type == CSGTree::NodeType::Cube)
    {
        return CubeSD(pos, primitivePos, primitiveParameters);
    }
    else if (type == CSGTree::NodeType::Cylinder)
    {
        return CylinderSD(pos, primitivePos, primitiveParameters);
    }
    return 0.0f;
}

__device__ void UnionSDWithIndex(FloatWithIndex a, FloatWithIndex& b)
{
    b = (a.f <= b.f) ? a : b;
}
__device__ void IntersectionSDWithIndex(FloatWithIndex a, FloatWithIndex& b)
{
    b = (a.f >= b.f) ? a : b;
}
__device__ void DifferenceSDWithIndex(FloatWithIndex a, FloatWithIndex& b)
{
    if (a.f >= -b.f)
    {
        b = a;
    }
    else
    {
        b.f = -b.f;
    }
}

__device__ void OperationWithIndex(int type, FloatWithIndex a, FloatWithIndex& b)
{
    if (type == CSGTree::NodeType::Union)
    {
        UnionSDWithIndex(a, b);
    }
    else if (type == CSGTree::NodeType::Intersection)
    {
        IntersectionSDWithIndex(a, b);
    }
    else if (type == CSGTree::NodeType::Difference)
    {
        DifferenceSDWithIndex(a, b);
    }
}

__device__ void UnionSD(float a, float& b)
{
    b = (a < b) ? a : b;
}
__device__ void IntersectionSD(float a, float& b)
{
    b = (a > b) ? a : b;
}
__device__ void DifferenceSD(float a, float& b)
{
    b = (a > -b) ? a : -b;
}

__device__ void Operation(int type, float a, float& b)
{
    if (type == CSGTree::NodeType::Union)
    {
        UnionSD(a, b);
    }
    else if (type == CSGTree::NodeType::Intersection)
    {
        IntersectionSD(a, b);
    }
    else if (type == CSGTree::NodeType::Difference)
    {
        DifferenceSD(a, b);
    }
}

inline __device__ RayHit GetDetailedHitInfo(
    Ray ray,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    FloatWithIndex mapResult,
    int t,
    int numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack,
    float3 camPos,
    bool inside
)
{
    RayHit hit;
    hit.hit = false;
    if (fabs(mapResult.f) > RAYMARCHEPSILON)
    {
        return hit;
    }
    hit.hit = true;
    hit.primitiveIdx = mapResult.idx;
    hit.position = ray.origin;
    hit.normal = SDNormal(ray, nodes, primitivePos, primitiveParameters, numberOfNodes, distanceStack);
    hit.normal = (inside ? -1 : 1) * hit.normal;
    return hit;
}

inline __device__ float3 SDNormal(
    Ray ray,
    CSGNode* nodes,
    CudaPrimitivePos* primitivePos,
    Parameters* primitiveParameters,
    int numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack
)
{
    float3 kxyy = make_float3(1, -1, -1),
        kyyx = make_float3(-1, -1, 1),
        kyxy = make_float3(-1, 1, -1),
        kxxx = make_float3(1, 1, 1);
    float first = MapCSGTree(
        nodes,
        primitivePos,
        primitiveParameters,
        ray.origin + RAYMARCH_H * kxyy,
        numberOfNodes, distanceStack);
    float second = MapCSGTree(
        nodes,
        primitivePos,
        primitiveParameters,
        ray.origin + RAYMARCH_H * kyyx,
        numberOfNodes, distanceStack);
    float third = MapCSGTree(
        nodes,
        primitivePos,
        primitiveParameters,
        ray.origin + RAYMARCH_H * kyxy,
        numberOfNodes, distanceStack);
    float fourth = MapCSGTree(
        nodes,
        primitivePos,
        primitiveParameters,
        ray.origin + RAYMARCH_H * kxxx,
        numberOfNodes, distanceStack);
    return normalize(first * kxyy +
        second * kyyx +
        third * kyxy +
        fourth * kxxx);
}
