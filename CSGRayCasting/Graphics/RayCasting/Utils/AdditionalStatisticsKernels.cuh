#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>

#include "CSGUtils.cuh"
#include "PrimitiveRayIntersection.cuh"
#include "../CSGTree/CSGTree.cuh"
#include "CudaCamera.cuh"

//
// ---- headers ----
//

__global__ void PrimitivePerPixelStatistic(
	CudaCamera cam,
	CSGNode* __restrict__ nodes,
	CudaPrimitivePos* __restrict__ primitivePos,
	Parameters* __restrict__ primitiveParameters,
	float width, float height,
	int primitiveCount,
	int* statistics);

//
// ---- code ----
//

__global__ void PrimitivePerPixelStatistic(
	CudaCamera cam,
	CSGNode* __restrict__ nodes,
	CudaPrimitivePos* __restrict__ primitivePos,
	Parameters* __restrict__ primitiveParameters,
	float width, float height,
	int primitiveCount,
	int* statistics)
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


	// Create ray from camera
	float3 rayOrigin = cam.position;
	float3 rayDirection = normalize(nx * cam.right + ny * cam.up + cam.forward);

	Ray ray(rayOrigin, rayDirection);
	RayHitMinimal hit;
	int hits = 0;
	int j = 0;
	for (int i = 0; j < primitiveCount; i++)
	{
		if (nodes[i].primitiveIdx != -1)
		{
			hitPrimitive(ray, primitivePos, primitiveParameters, nodes[i], hit, 0);
			if (hit.hit != CSG::CSGRayHit::Miss)
				hits++;
			j++;
		}
	}
	atomicAdd(statistics, hits);
	atomicAdd(&statistics[1], hits ? 1 : 0);
}