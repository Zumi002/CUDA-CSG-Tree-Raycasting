#pragma once

#include <cuda_runtime.h>

#include "../CSGTree.cuh"

class BVHNode
{
public:
	float minX = 0,
		minY = 0,
		minZ = 0;
	float maxX = 0,
		maxY = 0,
		maxZ = 0;

	__host__ __device__ BVHNode(Primitive prim, unsigned int type)
	{
		if (type == 3)
		{
			float r = prim.params.sphereParameters.radius;
			float x = prim.x;
			float y = prim.y;
			float z = prim.z;

			minX = x - r- r;
			minY = y - r - r;
			minZ = z - r - r;

			maxX = x + r + r;
			maxY = y + r + r;
			maxZ = z + r + r;
		}
		else if (type == 4)
		{
			//to do
			float maxR = std::max(prim.params.cylinderParameters.height / 2, prim.params.cylinderParameters.radius);
			float x = prim.x;
			float y = prim.y;
			float z = prim.z;

			minX = x - maxR;
			minY = y - maxR;
			minZ = z - maxR;

			maxX = x + maxR;
			maxY = y + maxR;
			maxZ = z + maxR;
		}
		else if (type == 5)
		{
			float halfSize = prim.params.cubeParameters.size / 2;
			float x = prim.x;
			float y = prim.y;
			float z = prim.z;

			minX = x - halfSize;
			minY = y - halfSize;
			minZ = z - halfSize;

			maxX = x + halfSize;
			maxY = y + halfSize;
			maxZ = z + halfSize;
		}
	}

	__host__ __device__ BVHNode(BVHNode Left, BVHNode right)
	{
		minX = std::min(Left.minX, right.minX);
		minY = std::min(Left.minY, right.minY);
		minZ = std::min(Left.minZ, right.minZ);

		maxX = std::max(Left.maxX, right.maxX);
		maxY = std::max(Left.maxY, right.maxY);
		maxZ = std::max(Left.maxZ, right.maxZ);
	}

	__host__ __device__ BVHNode()
	{
		minX = 0;
		minY = 0;
		minZ = 0;
		maxX = 0;
		maxY = 0;
		maxZ = 0;
	}
};