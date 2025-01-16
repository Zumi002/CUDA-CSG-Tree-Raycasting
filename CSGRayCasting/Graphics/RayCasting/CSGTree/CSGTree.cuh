#pragma once

#include <string>
#include <vector>
#include <sstream>
#include "Primitives/Primitives.h"
#include <stdexcept>
#include <stack>
#include <utility>

#include <cuda_runtime.h>

struct CSGNode
{
	int type;
	int primitiveIdx;
	int left;
	int right;
	int parent;

	__host__ __device__ CSGNode(int Type, int PrimitiveIdx, int Left, int Right, int Parent)
	{
		type = Type;
		primitiveIdx = PrimitiveIdx;
		left = Left;
		right = Right;
		parent = Parent;
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
		Cylinder
	};

	//node 0 is root of tree
	std::vector<CSGNode> nodes;

	Primitives primitives;

	static CSGTree Parse(const std::string& text);
};

struct CudaCSGTree
{
	CSGNode* nodes;

	Primitive* primitives;
};

std::vector<std::string> split(const std::string& text);
float color(const std::string& hex);