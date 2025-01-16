#pragma once

#include <cuda_runtime.h>
#include "CSGTree/CSGTree.cuh"
#include "Kernels/RaycastingKernels.cuh"

#define BLOCKXSIZE 8
#define BLOCKYSIZE 4

class Raycaster
{
	int width, height;
	dim3 blockDim;
	dim3 gridDim;
	
	CudaCSGTree cudaTree;
	RayHit* devHits;

	bool alloced = false;

	public:
		
		void ChangeSize(int newWidth, int newHeight, CSGTree tree);
		void Raycast(float4* devPBO, Camera cam);
};