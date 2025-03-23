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
#include "../CSGTree/BVH/BVHNode.cuh"

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

//BVH related kernels
__device__ bool isBVHNodeHit(const Ray& ray, const BVHNode& node, RayHitMinimal& hitInfo, float& tmin);



//kernel called once per pixel, returns ray intersection data in hits array
__global__ void RaycastKernel(Camera cam, CudaCSGTree tree, RayHit* hits, float width, float height);

//kernel called once per pixel, calculates output color based of ray intersecion data and lightnigng
__global__ void LightningKernel(Camera cam, RayHit* hits, Primitive* primitives, float4* output, float3 lightDir, float width, float height);

//decicdes which intersection method is called for given primitive node
__device__ void hitPrimitive(const Ray& ray, const CudaCSGTree& tree, const CSGNode& node, RayHitMinimal& hitInfo, float& tmin);

//functions which gets intersection data with given sphere
__device__ bool sphereHit(const Ray& ray, const Primitive& sphere, RayHitMinimal& hitInfo, float& tmin);

//functions which gets details about given intersection data  
__device__ void sphereHitDetails(const Ray& ray, const Primitive& sphere, const RayHitMinimal& hitInfo, RayHit& detailedHitInfo);

//same for cylinder
__device__ bool cylinderHit(const Ray& ray, const Primitive& cylinder, RayHitMinimal& hitInfo, float& tmin);

__device__ void cylinderHitDetails(const Ray& ray, const Primitive& cylinder, const RayHitMinimal& hitInfo, RayHit& detailedHitInfo);

//same for cube
__device__ bool cubeHit(const Ray& ray, const Primitive& sphere, RayHitMinimal& hitInfo, float& tmin);

__device__ void cubeHitDetails(const Ray& ray, const Primitive& sphere, const RayHitMinimal& hitInfo, RayHit& detailedHitInfo);

//CSG functions from paper [https://ceur-ws.org/Vol-1576/090.pdf]

//main loop of state machine
 inline __device__ void CSGRayCast(CudaCSGTree& tree, Ray& ray, RayHitMinimal& resultRayhit);

//function responsible of exploring tree
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
//function decides which action to take based of intersection of left and right subtree
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
//retrieves actions from predefined tables
 inline __device__ int LookUpActions(unsigned char lHit, unsigned char rHit, int op);

//returns parent node, or if not possible creates virtual node. If virutal node is created, tells state machine to end computations 
 inline __device__ CSGNode GetParent(CudaCSGTree& tree, CSGNode& node, bool& run);

 __global__ void CalculateInterscetion(int width, int height, int shape_count, CudaCSGTree dev_tree, int* parts, Camera cam,
	 RayHit* hits);

 __device__ float3 CalculateNormalVectorCylinder(const Primitive& cylinder, float3 pixelPosition);

 __device__ bool IntersectionPointSphere(const float3& spherePosition, float radius, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2);

 __device__ bool IntersectionPointCube(const Primitive& cube, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2, float3& N, float3& N2);

 __device__ bool IntersectionPointCylinder(const Primitive& cylinder, const float3& rayOrigin, const float3& rayDirection, float& t1, float& t2, float3& N, float3& N2);

 __device__ void AddIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);

 __host__ __device__ void AddIntervals2(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);
 __host__ __device__ void SubstractIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);
 __host__ __device__ void CommonPartIntervals(float* sphereIntersections, float* tempArray, int p1, int p2, int k1, int k2, bool print);

 __host__ __device__ float4 NormalizeVector4(float4 vector);

 __host__ __device__ float3 NormalizeVector3(float3 vector);

 __host__ __device__ float3 cross(const float3& a, const float3& b);

 __host__ __device__ float dot3(const float3& a, const float3& b);

 __host__ __device__ void MultiplyVectorByMatrix4(float4& vector, const float* matrix);