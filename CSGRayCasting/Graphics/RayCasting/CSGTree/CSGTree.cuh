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

struct CSGNode
{
	int type;
	int primitiveIdx;
	int left;
	int right;
	int parent;

	BVHNode bvhNode;

	__host__ __device__ CSGNode(int Type, int PrimitiveIdx, int Left, int Right, int Parent)
	{
		type = Type;
		primitiveIdx = PrimitiveIdx;
		left = Left;
		right = Right;
		parent = Parent;
		bvhNode = BVHNode();
	}
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
	void ConstructBVH();
};

struct CudaCSGTree
{
	CSGNode* nodes;

	Primitive* primitives;
};

std::vector<std::string> split(const std::string& text);
float color(const std::string& hex);