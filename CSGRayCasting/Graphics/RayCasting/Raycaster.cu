#include "Raycaster.cuh"

void Raycaster::ChangeSize(int newWidth, int newHeight, CSGTree tree)
{
	if (alloced)
	{
		gpuErrchk(cudaFree(devHits));
		gpuErrchk(cudaFree(cudaTree.nodes));
		gpuErrchk(cudaFree(cudaTree.primitives));
	}

	width = newWidth;
	height = newHeight;

	blockDim = dim3(BLOCKXSIZE, BLOCKYSIZE);
	gridDim = dim3((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

	gpuErrchk(cudaMalloc(&cudaTree.nodes, tree.nodes.size() * sizeof(CSGNode)));
	gpuErrchk(cudaMalloc(&cudaTree.primitives, tree.primitives.primitives.size() * sizeof(Primitive)));

	gpuErrchk(cudaMemcpy(cudaTree.nodes, tree.nodes.data(), tree.nodes.size() * sizeof(CSGNode), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(cudaTree.primitives, tree.primitives.primitives.data(), tree.primitives.primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMalloc(&devHits, width * height * sizeof(RayHit)));
	alloced = true;
}

void Raycaster::Raycast(float4* devPBO, Camera cam)
{

	RaycastKernel<<<gridDim, blockDim>>>(cam, cudaTree, devHits, width, height);
	cudaDeviceSynchronize();

	LightningKernel<<<gridDim, blockDim>>>(cam, devHits, cudaTree.primitives, devPBO, width, height);
	cudaDeviceSynchronize();
	
}

