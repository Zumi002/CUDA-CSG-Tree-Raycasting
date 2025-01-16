#pragma once

#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cmath>
#include <math.h>

#include "../../RenderManager/Camera/Camera.h"
#include "../CSGTree/CSGTree.cuh"
#include "../Utils/Ray.cuh"
#include "../Utils/CudaStack.cuh"
#include "../Utils/CSGUtils.cuh"
#include "../Utils/Float3Utils.cuh"
#include "../../RenderManager/DirectionalLight.h"

#define MAXSTACKSIZE 32

//macro from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort)
			exit(code);
	}
}

__global__ void RaycastKernel(Camera cam, CudaCSGTree tree, RayHit* hits, float width, float height);

__global__ void LightningKernel(Camera cam, RayHit* hits, Primitive* primitives, float4* output, float3 lightDir, float width, float height);

__device__ bool sphereHit(const Ray& ray, const Primitive& sphere, RayHitMinimal& hitInfo, float& tmin);

__device__ void sphereHitDetails(const Ray& ray, const Primitive& sphere, const RayHitMinimal& hitInfo, RayHit& detailedHitInfo);



//CSG functions from paper [https://ceur-ws.org/Vol-1576/090.pdf]

 inline __device__ void CSGRayCast(CudaCSGTree& tree, Ray& ray, RayHitMinimal& resultRayhit);
 inline __device__ void GoTo(
	CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
	CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
	CudaStack<float, MAXSTACKSIZE>& timeStack,
	unsigned char& action,
	CSGNode& node,
	CudaCSGTree& tree,
	RayHitMinimal& leftRay,
	RayHitMinimal& rightRay,
	Ray& ray,
	float& tmin,
	bool& run);
 inline __device__ void Compute(
		CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
		CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
		CudaStack<float, MAXSTACKSIZE>& timeStack,
		unsigned char& action,
		CSGNode& node,
		CudaCSGTree& tree,
		RayHitMinimal& leftRay,
		RayHitMinimal& rightRay,
		float& tmin,
		bool& run);
 inline __device__ int LookUpActions(unsigned char lHit, unsigned char rHit, int op);
 inline __device__ CSGNode GetParent(CudaCSGTree& tree, CSGNode& node, bool& run);

