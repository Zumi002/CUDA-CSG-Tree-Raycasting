#pragma once

#include <cuda_runtime.h>

#include "../CSGTree.cuh"

struct BVHNode
{

	float minX = 0,
		minY = 0,
		minZ = 0;
	float maxX = 0,
		maxY = 0,
		maxZ = 0;

	__host__ __device__ BVHNode(CudaPrimitivePos pos, Parameters params, unsigned int type)
	{
		if (type == 3)
		{
			float r = params.sphereParameters.radius;
			float x = pos.x;
			float y = pos.y;
			float z = pos.z;

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
			float maxR = fmax(params.cylinderParameters.height / 2, params.cylinderParameters.radius);
			float x = pos.x;
			float y = pos.y;
			float z = pos.z;

			minX = x - maxR;
			minY = y - maxR;
			minZ = z - maxR;

			maxX = x + maxR;
			maxY = y + maxR;
			maxZ = z + maxR;
		}
		else if (type == 5)
		{
			float halfSize = params.cubeParameters.size / 2;
			float x = pos.x;
			float y = pos.y;
			float z = pos.z;

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
		minX = fmin(Left.minX, right.minX);
		minY = fmin(Left.minY, right.minY);
		minZ = fmin(Left.minZ, right.minZ);

		maxX = fmax(Left.maxX, right.maxX);
		maxY = fmax(Left.maxY, right.maxY);
		maxZ = fmax(Left.maxZ, right.maxZ);
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