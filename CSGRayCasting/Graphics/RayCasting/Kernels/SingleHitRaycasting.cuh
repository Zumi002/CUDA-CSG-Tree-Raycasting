#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../Utils/PrimitiveRayIntersection.cuh"
#include "../Utils/CudaStack.cuh"
#include "../CSGTree/CSGTree.cuh"
#include "../Utils/CudaCamera.cuh"

#define MAXSTACKSIZE 32

//
// ---- headers ----
//

//function checks if ray intersects with given node
inline __device__ bool isBVHNodeHit(Ray ray, BVHNode node, RayHitMinimal& hitInfo, float tmin);

//kernel called once per pixel, returns ray intersection data in hits array
__global__ void RaycastKernel(
	CudaCamera cam,
	CSGNode* __restrict__ nodes,
	BVHNode* __restrict__ bvhNodes,
	CudaPrimitivePos* __restrict__ primitivePos,
	Parameters* __restrict__ primitiveParameters,
	RayHit* __restrict__ hits,
	float width, float height
);

//CSG functions from paper [https://ceur-ws.org/Vol-1576/090.pdf]

//main loop of state machine
inline __device__ void CSGRayCast(
	const CSGNode* __restrict__ nodes,
	const CudaPrimitivePos* __restrict__ primitivePos,
	const Parameters* __restrict__  primitiveParameters,
	BVHNode* __restrict__ bvhNodes,
	Ray& ray, RayHitMinimal& resultRayhit
);

//function responsible of exploring tree
inline __device__ void GoTo(
	CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
	CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
	CudaStack<float, MAXSTACKSIZE>& timeStack,
	unsigned char& action,
	CSGNode& node,
	const CSGNode* __restrict__ nodes,
	const CudaPrimitivePos* __restrict__ primitivePos,
	const Parameters* __restrict__ primitiveParameters,
	BVHNode* __restrict__ bvhNodes,
	RayHitMinimal& leftRay,
	RayHitMinimal& rightRay,
	Ray& ray,
	float& tmin,
	bool& run
);

//function decides which action to take based of intersection of left and right subtree
inline __device__ void Compute(
	CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
	CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
	CudaStack<float, MAXSTACKSIZE>& timeStack,
	unsigned char& action,
	CSGNode& node,
	const CSGNode* __restrict__ nodes,
	const CudaPrimitivePos* __restrict__ primitivePos,
	const Parameters* __restrict__ primitiveParameters,
	RayHitMinimal& leftRay,
	RayHitMinimal& rightRay,
	float& tmin,
	bool& run
);

//retrieves actions from predefined tables
inline __device__ int LookUpActions(unsigned char lHit, unsigned char rHit, int op);

//returns parent node, or if not possible creates virtual node. If virutal node is created, tells state machine to end computations 
inline __device__ CSGNode GetParent(const CSGNode* __restrict__ nodes, CSGNode& node, bool& run);

//
// ---- code ----
//

__global__ void RaycastKernel(
	CudaCamera  cam,
	CSGNode* __restrict__ nodes,
	BVHNode* __restrict__ bvhNodes,
	CudaPrimitivePos* __restrict__ primitivePos,
	Parameters* __restrict__ primitiveParameters,
	RayHit* __restrict__ hits,
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


	// Create ray from camera
	float3 rayOrigin = cam.position;
	float3 rayDirection = normalize(nx * cam.right + ny * cam.up + cam.forward);

	Ray ray(rayOrigin, rayDirection);
	RayHitMinimal hitInfo;

	CSGRayCast(nodes, primitivePos, primitiveParameters, bvhNodes, ray, hitInfo);
	int pixelIdx = (y * (int)width + x);

	RayHit detailedHitInfo;
	detailedHitInfo.hit = false;
	if (hitInfo.hit != CSG::CSGRayHit::Miss)
	{
		if (hitInfo.primitiveType == CSGTree::NodeType::Sphere)
			sphereHitDetails(ray, primitivePos[hitInfo.primitiveIdx], primitiveParameters[hitInfo.primitiveIdx], hitInfo, detailedHitInfo);
		if (hitInfo.primitiveType == CSGTree::NodeType::Cylinder)
			cylinderHitDetails(ray, primitivePos[hitInfo.primitiveIdx], primitiveParameters[hitInfo.primitiveIdx], hitInfo, detailedHitInfo);
		if (hitInfo.primitiveType == CSGTree::NodeType::Cube)
			cubeHitDetails(ray, primitivePos[hitInfo.primitiveIdx], primitiveParameters[hitInfo.primitiveIdx], hitInfo, detailedHitInfo);
	}
	hits[pixelIdx] = detailedHitInfo;
}

inline __device__ void CSGRayCast(
	const CSGNode* __restrict__ nodes,
	const CudaPrimitivePos* __restrict__ primitivePos,
	const Parameters* __restrict__ primitiveParameters,
	BVHNode* __restrict__ bvhNodes,
	Ray& ray, RayHitMinimal& resultRayhit
)
{

	CudaStack<unsigned char, MAXSTACKSIZE> actionStack;
	CudaStack<RayHitMinimal, MAXSTACKSIZE> primitiveStack;
	CudaStack<float, MAXSTACKSIZE> timeStack;

	float tmin = 0;
	CSGNode node = CSGNode(0, 0, 0, 0, 0);
	RayHitMinimal leftRay;
	leftRay.hit = CSG::CSGRayHit::Miss;
	RayHitMinimal rightRay;
	rightRay.hit = CSG::CSGRayHit::Miss;
	actionStack.push(CSG::CSGActions::Compute);
	unsigned char action = CSG::CSGActions::GotoLft;
	bool run = true;

	while (run || actionStack.size() > 0)
	{
		if (action & CSG::CSGActions::SaveLft)
		{
			tmin = timeStack.pop();
			primitiveStack.push(leftRay);
			action = CSG::CSGActions::GotoRgh;
		}
		if (action & (CSG::CSGActions::GotoLft | CSG::CSGActions::GotoRgh))
		{
			GoTo(actionStack,
				primitiveStack,
				timeStack,
				action,
				node,
				nodes,
				primitivePos,
				primitiveParameters,
				bvhNodes,
				leftRay,
				rightRay,
				ray,
				tmin,
				run);
		}
		if (action & (CSG::CSGActions::LoadLft | CSG::CSGActions::LoadRgh | CSG::CSGActions::Compute))
		{
			Compute(actionStack,
				primitiveStack,
				timeStack,
				action,
				node,
				nodes,
				primitivePos,
				primitiveParameters,
				leftRay,
				rightRay,
				tmin,
				run);
		}
	}

	resultRayhit = leftRay;
}

inline __device__ void GoTo(
	CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
	CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
	CudaStack<float, MAXSTACKSIZE>& timeStack,
	unsigned char& action,
	CSGNode& node,
	const CSGNode* __restrict__ nodes,
	const CudaPrimitivePos* __restrict__ primitivePos,
	const Parameters* __restrict__ primitiveParameters,
	BVHNode* __restrict__ bvhNodes,
	RayHitMinimal& leftRay,
	RayHitMinimal& rightRay,
	Ray& ray,
	float& tmin,
	bool& run)
{
	if (action & CSG::CSGActions::GotoLft)
	{
		node = nodes[node.left];
	}
	else
	{
		node = nodes[node.right];
	}

	if (node.type == CSGTree::NodeType::Union ||
		node.type == CSGTree::NodeType::Difference ||
		node.type == CSGTree::NodeType::Intersection)
	{
		bool gotoL = isBVHNodeHit(ray, bvhNodes[node.left], leftRay, tmin);
		bool gotoR = isBVHNodeHit(ray, bvhNodes[node.right], rightRay, tmin);
		CSGNode tmpNode = nodes[node.left];
		if (gotoL && (tmpNode.primitiveIdx != -1))
		{
			hitPrimitive(ray, primitivePos, primitiveParameters, tmpNode, leftRay, tmin);
			gotoL = false;
		}
		tmpNode = nodes[node.right];
		if (gotoR && (nodes[node.right].primitiveIdx != -1))
		{
			hitPrimitive(ray, primitivePos, primitiveParameters, tmpNode, rightRay, tmin);
			gotoR = false;
		}
		if (gotoL || gotoR)
		{
			if (!gotoL)
			{
				primitiveStack.push(leftRay);
				actionStack.push(CSG::CSGActions::LoadLft);
				action = CSG::CSGActions::GotoRgh;
			}
			else if (!gotoR)
			{
				primitiveStack.push(rightRay);
				actionStack.push(CSG::CSGActions::LoadRgh);
				action = CSG::CSGActions::GotoLft;
			}
			else
			{
				timeStack.push(tmin);
				actionStack.push(CSG::CSGActions::LoadLft);
				actionStack.push(CSG::CSGActions::SaveLft);
				action = CSG::CSGActions::GotoLft;
			}
		}
		else
		{
			action = CSG::CSGActions::Compute;
		}

	}
	else
	{
		if (action & CSG::CSGActions::GotoLft)
		{
			hitPrimitive(ray, primitivePos, primitiveParameters, node, leftRay, tmin);
		}
		else
		{
			hitPrimitive(ray, primitivePos, primitiveParameters, node, rightRay, tmin);
		}
		action = actionStack.pop();
		node = GetParent(nodes, node, run);
	}
}

inline __device__ void Compute(
	CudaStack<unsigned char, MAXSTACKSIZE>& actionStack,
	CudaStack<RayHitMinimal, MAXSTACKSIZE>& primitiveStack,
	CudaStack<float, MAXSTACKSIZE>& timeStack,
	unsigned char& action,
	CSGNode& node,
	const CSGNode* __restrict__ nodes,
	const CudaPrimitivePos* __restrict__ primitivePos,
	const Parameters* __restrict__ primitiveParameters,
	RayHitMinimal& leftRay,
	RayHitMinimal& rightRay,
	float& tmin,
	bool& run)
{
	if (action & (CSG::CSGActions::LoadLft | CSG::CSGActions::LoadRgh))
	{
		if (action & CSG::CSGActions::LoadLft)
		{
			leftRay = primitiveStack.pop();
		}
		else
		{
			rightRay = primitiveStack.pop();
		}
	}
	int actions = LookUpActions(leftRay.hit, rightRay.hit, node.type);
	if ((actions & CSG::HitActions::RetL) || ((actions & CSG::HitActions::RetLIfCloser) && (leftRay.t <= rightRay.t)))
	{
		rightRay = leftRay;
		action = actionStack.pop();
		node = GetParent(nodes, node, run);
	}
	else if ((actions & CSG::HitActions::RetR) || ((actions & CSG::HitActions::RetRIfCloser) && (leftRay.t > rightRay.t)))
	{
		if (actions & CSG::HitActions::FlipR)
		{
			rightRay.hit ^= CSG::CSGRayHit::Flip;
			rightRay.hit ^= CSG::CSGRayHit::Exit;
			rightRay.hit ^= CSG::CSGRayHit::Enter;

		}
		leftRay = rightRay;
		action = actionStack.pop();
		node = GetParent(nodes, node, run);
	}
	else if ((actions & CSG::HitActions::LoopL) || ((actions & CSG::HitActions::LoopLIfCloser) && (leftRay.t <= rightRay.t)))
	{
		tmin = leftRay.t;
		primitiveStack.push(rightRay);
		actionStack.push(CSG::CSGActions::LoadRgh);
		action = CSG::CSGActions::GotoLft;
	}
	else if ((actions & CSG::HitActions::LoopR) || ((actions & CSG::HitActions::LoopRIfCloser) && (leftRay.t > rightRay.t)))
	{
		tmin = rightRay.t;
		primitiveStack.push(leftRay);
		actionStack.push(CSG::CSGActions::LoadLft);
		action = CSG::CSGActions::GotoRgh;
	}
	else
	{
		rightRay = RayHitMinimal();
		rightRay.hit = CSG::CSGRayHit::Miss;
		leftRay = RayHitMinimal();
		leftRay.hit = CSG::CSGRayHit::Miss;
		action = actionStack.pop();
		node = GetParent(nodes, node, run);
	}
}

__constant__ int unionTable[3][3] = {
	{{CSG::HitActions::RetLIfCloser | CSG::HitActions::RetRIfCloser},{CSG::HitActions::RetRIfCloser | CSG::HitActions::LoopL},{CSG::HitActions::RetL}},
	{{CSG::HitActions::RetLIfCloser | CSG::HitActions::LoopR},{CSG::HitActions::LoopLIfCloser | CSG::HitActions::LoopRIfCloser},{CSG::HitActions::RetL}},
	{{CSG::HitActions::RetR},{CSG::HitActions::RetR},{CSG::HitActions::MissAction}} };
__constant__ int intersectionTable[3][3] = {
{{CSG::HitActions::LoopLIfCloser | CSG::HitActions::LoopRIfCloser},{CSG::HitActions::RetLIfCloser | CSG::HitActions::LoopR},{CSG::HitActions::MissAction}},
{{CSG::HitActions::RetRIfCloser | CSG::HitActions::LoopL},{CSG::HitActions::RetLIfCloser | CSG::HitActions::RetRIfCloser},{CSG::HitActions::MissAction}},
{{CSG::HitActions::MissAction},{CSG::HitActions::MissAction},{CSG::HitActions::MissAction}} };
__constant__ int differenceTable[3][3] = {
{{CSG::HitActions::RetLIfCloser | CSG::HitActions::LoopR},{CSG::HitActions::LoopLIfCloser | CSG::HitActions::LoopRIfCloser},{CSG::HitActions::RetL}},
{{CSG::HitActions::RetLIfCloser | CSG::HitActions::RetRIfCloser | CSG::HitActions::FlipR},{CSG::HitActions::RetRIfCloser | CSG::HitActions::FlipR | CSG::HitActions::LoopL},{CSG::HitActions::RetL}},
{{CSG::HitActions::MissAction},{CSG::HitActions::MissAction},{CSG::HitActions::MissAction}} };

inline __device__ int LookUpActions(unsigned char lHit, unsigned char rHit, int op)
{

	if (lHit & CSG::CSGRayHit::Enter)
		lHit = 0;
	else if (lHit & CSG::CSGRayHit::Exit)
		lHit = 1;
	else if (lHit & CSG::CSGRayHit::Miss)
		lHit = 2;

	if (rHit & CSG::CSGRayHit::Enter)
		rHit = 0;
	else if (rHit & CSG::CSGRayHit::Exit)
		rHit = 1;
	else if (rHit & CSG::CSGRayHit::Miss)
		rHit = 2;

	if (op == CSGTree::NodeType::Union)
	{
		return unionTable[lHit][rHit];
	}
	else if (op == CSGTree::NodeType::Intersection)
	{
		return intersectionTable[lHit][rHit];
	}
	else if (op == CSGTree::NodeType::Difference)
	{
		return differenceTable[lHit][rHit];
	}
	return -1;
}

inline __device__ CSGNode GetParent(const CSGNode* __restrict__ nodes, CSGNode& node, bool& run)
{
	if (node.parent >= 0)
	{
		return nodes[node.parent];
	}
	run = false;
	return CSGNode(0, 0, 0, 0, 0);
}

inline __device__ bool isBVHNodeHit(Ray ray, BVHNode node, RayHitMinimal& hitInfo, float tmin)
{


	float3 lb = make_float3(node.minX, node.minY, node.minZ);
	float3 rt = make_float3(node.maxX, node.maxY, node.maxZ);
	float t1 = (lb.x - ray.origin.x) / ray.direction.x;
	float t2 = (rt.x - ray.origin.x) / ray.direction.x;
	float t3 = (lb.y - ray.origin.y) / ray.direction.y;
	float t4 = (rt.y - ray.origin.y) / ray.direction.y;
	float t5 = (lb.z - ray.origin.z) / ray.direction.z;
	float t6 = (rt.z - ray.origin.z) / ray.direction.z;

	float tempmin = fmax(fmax(fmin(t1, t2), fmin(t3, t4)), fmin(t5, t6));
	float tempmax = fmin(fmin(fmax(t1, t2), fmax(t3, t4)), fmax(t5, t6));

	// if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind us
	if (tempmax < 0)
	{
		hitInfo.hit = CSG::CSGRayHit::Miss;
		return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tempmin > tempmax)
	{
		hitInfo.hit = CSG::CSGRayHit::Miss;
		return false;
	}
	if (tempmin <= tmin)
	{
		if (tempmax <= tmin)
		{
			hitInfo.hit = CSG::CSGRayHit::Miss;
			return false;
		}
	}

	return true;
}
