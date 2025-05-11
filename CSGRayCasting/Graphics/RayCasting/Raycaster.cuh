#pragma once

#include <cuda_runtime.h>
#include "CSGTree/CSGTree.cuh"
#include "../RenderManager/DirectionalLight.h"
#include "../RenderManager/Camera/Camera.h"
#include "Utils/RayHit.cuh"
#include "Utils/CudaCamera.cuh"
#include <cudaProfiler.h>
#include <cuda_profiler_api.h>

#define BLOCKXSIZE 8
#define BLOCKYSIZE 4

#define BLOCKXSIZECLASSIC 8
#define BLOCKYSIZECLASSIC 4

#define BLOCKXSIZERAYMARCH 16
#define BLOCKYSIZERAYMARCH 8

#define BLOCKXSIZELIGHTING 16
#define BLOCKYSIZELIGHTING 16

class Raycaster
{
	int width, height;

	dim3 blockDimSingle;
	dim3 gridDimSingle;

	dim3 blockDimClassic;
	dim3 gridDimClassic;

	dim3 blockDimRayMarch;
	dim3 gridDimRayMarch;

	dim3 blockDimLighting;
	dim3 gridDimLighting;

	CudaCSGTree cudaTree;
	RayHit* devHits;
	CudaCamera* devCamera;
	
	BVHNode* devBvhNodes;
	int* devParts;
	int* Parts;
	int alg = 0;
	int shapeCount = 0;
	int nodeCount = 0;
	CudaCamera cudaCamera = CudaCamera();

	bool alloced = false;
	bool allocedTree = false;
	bool allocedClassicalAdds = false;
	bool allocedSingleHitAdds = false;

	void MapFromCamera(Camera cam);

public:
	Raycaster();
	void ChangeTree(CSGTree& tree);
	void ChangeSize(int newWidth, int newHright);
	void ChangeSize(int newWidth, int newHeight, CSGTree& tree);
	void Raycast(float4* devPBO, Camera cam, DirectionalLight light);
	void CleanUpTree();
	void CleanUpTexture();
	void CleanUp();
	void ChangeAlg(CSGTree& tree, int alg);
	void CleanUpClassical();
	void SetupClassical(CSGTree& tree);
	void SetupSingleHit(CSGTree& tree);
	void CleanUpSingleHit();
};