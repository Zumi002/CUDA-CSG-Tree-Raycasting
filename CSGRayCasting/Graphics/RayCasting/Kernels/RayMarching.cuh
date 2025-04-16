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

__device__ inline FloatWithIndex MapCSGTreeWithIndex(
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const float3& ray,
    const int& numberOfNodes,
    CudaStack<FloatWithIndex, MAXRAYMARCHINGSTACKSIZE>& distanceStack
);

__device__ inline float MapCSGTree(
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const float3& ray,
    const int& numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack
);

__device__ inline float GetSD(
    const CudaPrimitivePos& primitivePos,
    const Parameters& primitiveParameters,
    const int& type,
    const float3& ray
);

__device__ inline void UnionSDWithIndex(
    const FloatWithIndex& a,
    FloatWithIndex& b
);
__device__ inline void IntersectionSDWithIndex(
    const FloatWithIndex& a,
    FloatWithIndex& b
);
__device__ inline void DifferenceSDWithIndex(
    const FloatWithIndex& a,
    FloatWithIndex& b
);

__device__ inline void OperationWithIndex(
    const int& type,
    const FloatWithIndex& a,
    FloatWithIndex& b
);

__device__ inline void UnionSD(
    const float& a,
    float& b
);
__device__ inline void IntersectionSD(
    const float& a,
    float& b
);
__device__ inline void DifferenceSD(
    const float& a,
    float& b
);

__device__ inline void Operation(
    const int& type,
    const float& a,
    float& b
);

__device__ inline RayHit GetDetailedHitInfo(
    const Ray& ray,
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const FloatWithIndex& nodeIdx, 
    const int& t,
    const int& numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack,
    const float3& camPos   
);

__device__ inline float3 SDNormal(
    const Ray& ray,
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const int& numberOfNodes,
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

    CudaStack<float, MAXRAYMARCHINGSTACKSIZE> distanceStack;

    int marches = 0;
    float mapResult = RAYMARCHFAR;
    float t = 0;
    int minDistPrimitiveIdx = -1;
    while (marches < MAXMARCHES && fabs(mapResult) > RAYMARCHEPSILON && t < RAYMARCHFAR)
    {
        mapResult = MapCSGTree(nodes, primitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStack);
        ray.origin = ray.origin + fabs(mapResult) * ray.direction;
        t += fabs(mapResult);
        marches++;
    }
    CudaStack<FloatWithIndex, MAXRAYMARCHINGSTACKSIZE> distanceStackWithIndex;
    FloatWithIndex mapResultWithIdx = MapCSGTreeWithIndex(nodes, primitivePos, primitiveParameters, ray.origin, numberOfNodes, distanceStackWithIndex);

    hits[pixelIdx] = GetDetailedHitInfo(ray, nodes, primitivePos, primitiveParameters, mapResultWithIdx, t, numberOfNodes, distanceStack, cam.position);
}

__device__ FloatWithIndex MapCSGTreeWithIndex(
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const float3& pos,
    const int& numberOfNodes,
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
        if (nodes[i].primitiveIdx == -1)
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
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const float3& pos,
    const int& numberOfNodes,
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
        if (nodes[i].primitiveIdx == -1)
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
    const CudaPrimitivePos& primitivePos,
    const Parameters& primitiveParameters,
    const int& type,
    const float3& pos
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

__device__ void UnionSDWithIndex(const FloatWithIndex& a, FloatWithIndex& b)
{
    b = (a.f < b.f) ? a : b;
}
__device__ void IntersectionSDWithIndex(const FloatWithIndex& a, FloatWithIndex& b)
{
    b = (a.f > b.f) ? a : b;
}
__device__ void DifferenceSDWithIndex(const FloatWithIndex& a, FloatWithIndex& b)
{
    if (a.f > -b.f)
    {
        b = a;
    }
    else
    {
        b.f = -b.f;
    }
}

__device__ void OperationWithIndex(const int& type, const FloatWithIndex& a, FloatWithIndex& b)
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

__device__ void UnionSD(const float& a, float& b)
{
    b = (a < b) ? a : b;
}
__device__ void IntersectionSD(const float& a, float& b)
{
    b = (a > b) ? a : b;
}
__device__ void DifferenceSD(const float& a, float& b)
{
    b = (a > -b) ? a : -b;
}

__device__ void Operation(const int& type, const float& a, float& b)
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
    const Ray& ray,
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const FloatWithIndex& mapResult, 
    const int& t,
    const int& numberOfNodes,
    CudaStack<float, MAXRAYMARCHINGSTACKSIZE>& distanceStack,
    const float3& camPos
)
{
    RayHit hit;
    hit.hit = false;
    if (fabs(mapResult.f) > RAYMARCHEPSILON)
    {
        return hit;
    }
    hit.hit = true;
    hit.t = t;
    hit.primitiveIdx = mapResult.idx;
    hit.position = ray.origin;
    hit.normal = SDNormal(ray, nodes, primitivePos, primitiveParameters, numberOfNodes, distanceStack);
    return hit;
}

inline __device__ float3 SDNormal(
    const Ray& ray,
    const CSGNode*& nodes,
    const CudaPrimitivePos*& primitivePos,
    const Parameters*& primitiveParameters,
    const int& numberOfNodes,
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
        ray.origin + RAYMARCH_H* kxyy,
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
    return normalize(first*kxyy +
                     second*kyyx +
                     third*kyxy +
                     fourth*kxxx);
}