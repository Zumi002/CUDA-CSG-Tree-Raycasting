#pragma once

#include <cuda_runtime.h>
#include "CSGTree/CSGTree.cuh"
#include "../RenderManager/DirectionalLight.h"
#include "../RenderManager/Camera/Camera.h"
#include "Utils/RayHit.cuh"
#include "Utils/CudaCamera.cuh"

#define BLOCKXSIZE 8
#define BLOCKYSIZE 4

class Raycaster
{
	int width, height;
	dim3 blockDim;
	dim3 gridDim;

	CudaCSGTree cudaTree;
	RayHit* devHits;
	int* devParts;
	int* Parts;
	int alg = 0;
	int shapeCount = 0;
	CudaCamera cudaCamera = CudaCamera();

	bool alloced = false;
	bool allocedTree = false;
	bool allocedClassicalAdds = false;

	void MapFromCamera(Camera cam);

public:
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

};