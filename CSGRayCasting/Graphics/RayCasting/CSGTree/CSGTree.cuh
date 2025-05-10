#pragma once

#include <string>
#include <vector>
#include <sstream>
#include "Primitives/Primitives.h"
#include <stdexcept>
#include <stack>
#include <utility>

#include "BVH/BVHNode.cuh"

#include <cuda_runtime.h>

struct __align__(16) CSGNode
{
	short type;
	short primitiveIdx;
	int left;
	int right;
	int parent;

	__host__ __device__ CSGNode(short Type, short PrimitiveIdx, int Left, int Right, int Parent)
	{
		type = Type;
		primitiveIdx = PrimitiveIdx;
		left = Left;
		right = Right;
		parent = Parent;
	}

	__device__ CSGNode() {}
};

struct CSGTree
{
	enum NodeType
	{
		Union,
		Difference,
		Intersection,
		Sphere,
		Cylinder,
		Cube
	};

	//node 0 is root of tree
	std::vector<CSGNode> nodes;

	Primitives primitives;

	static CSGTree Parse(const std::string& text);
	std::vector<BVHNode> ConstructBVH();
	void TransformForClassical();
	void TransformForRaymarching();
};

struct CudaCSGTree
{
	CSGNode* nodes;
	CudaPrimitivePos* primitivePos;
	CudaPrimitiveColor* primitiveColor;
	Parameters* primitiveParams;
};

std::vector<std::string> split(const std::string& text);
float color(const std::string& hex);
void CreateParts(CSGTree& tree, int* parts, int node);
