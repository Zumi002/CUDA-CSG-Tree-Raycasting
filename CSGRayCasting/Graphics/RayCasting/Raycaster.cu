#include "Raycaster.cuh"
#include "Kernels/Kernels.cuh"
#include "cuda_profiler_api.h"


void Raycaster::ChangeTree(CSGTree& tree)
{
    CleanUpTree();
    gpuErrchk(cudaMalloc(&cudaTree.nodes, tree.nodes.size() * sizeof(CSGNode)));
    gpuErrchk(cudaMalloc(&cudaTree.primitivePos, tree.primitives.primitivePos.size() * sizeof(CudaPrimitivePos)));
    gpuErrchk(cudaMalloc(&cudaTree.primitiveColor, tree.primitives.primitiveColor.size() * sizeof(CudaPrimitiveColor)));
    gpuErrchk(cudaMalloc(&cudaTree.primitiveParams, tree.primitives.primitiveParameters.size() * sizeof(Parameters)));

    gpuErrchk(cudaMemcpy(cudaTree.nodes, tree.nodes.data(), tree.nodes.size() * sizeof(CSGNode), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudaTree.primitivePos, tree.primitives.primitivePos.data(), tree.primitives.primitivePos.size() * sizeof(CudaPrimitivePos), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudaTree.primitiveColor, tree.primitives.primitiveColor.data(), tree.primitives.primitiveColor.size() * sizeof(CudaPrimitiveColor), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudaTree.primitiveParams, tree.primitives.primitiveParameters.data(), tree.primitives.primitiveParameters.size() * sizeof(Parameters), cudaMemcpyHostToDevice));
    allocedTree = true;
    nodeCount = tree.nodes.size();
}

void Raycaster::ChangeSize(int newWidth, int newHeight)
{
    CleanUpTexture();
    width = newWidth;
    height = newHeight;
    gpuErrchk(cudaMalloc(&devHits, width * height * sizeof(RayHit)));
    alloced = true;
}

void Raycaster::ChangeSize(int newWidth, int newHeight, CSGTree& tree)
{
    ChangeSize(newWidth, newHeight);
    ChangeTree(tree);
}

void Raycaster::Raycast(float4* devPBO, Camera cam, DirectionalLight light)
{
    MapFromCamera(cam);
    cudaProfilerStart();
    if (alg == 0)
        RaycastKernel << <gridDim, blockDim >> > (cudaCamera, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devHits, width, height);
    else if (alg == 1)
        CalculateInterscetion << <gridDim, blockDim >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
    else if (alg == 2)
        RaymarchingKernel << <gridDim, blockDim >> > (cudaCamera, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devHits, nodeCount, width, height);
    cudaDeviceSynchronize();



    LightningKernel << <gridDim, blockDim >> > (cudaCamera, devHits, cudaTree.primitiveColor, devPBO, light.getLightDir(), width, height);
    cudaDeviceSynchronize();
    cudaProfilerStop();
}

void Raycaster::CleanUpTree()
{
    if (allocedTree)
    {
        gpuErrchk(cudaFree(cudaTree.nodes));
        gpuErrchk(cudaFree(cudaTree.primitivePos));
        gpuErrchk(cudaFree(cudaTree.primitiveColor));
        gpuErrchk(cudaFree(cudaTree.primitiveParams));
        allocedTree = false;
    }
}
void Raycaster::CleanUpTexture()
{
    if (alloced)
    {
        gpuErrchk(cudaFree(devHits));
        alloced = false;
    }
}

void Raycaster::CleanUpClassical()
{
    if (allocedClassicalAdds)
    {
        gpuErrchk(cudaFree(devParts));
        free(Parts);
        allocedClassicalAdds = false;
    }

}

void  Raycaster::SetupClassical(CSGTree& tree)
{
    if (allocedTree&&tree.nodes.size())
    {
        Parts = (int*)malloc(tree.primitives.primitivePos.size() * 4 * sizeof(int));
        CreateParts(tree, Parts, 0);
        gpuErrchk(cudaMalloc(&devParts, tree.primitives.primitivePos.size() * 4 * sizeof(int)));
        gpuErrchk(cudaMemcpy(devParts, Parts, tree.primitives.primitivePos.size() * 4 * sizeof(int), cudaMemcpyHostToDevice));
        shapeCount = tree.primitives.primitivePos.size();
        allocedClassicalAdds = true;
    }
}

void Raycaster::CleanUp()
{
    CleanUpTree();
    CleanUpTexture();
    CleanUpClassical();
}

void Raycaster::ChangeAlg(CSGTree& tree, int newAlg)
{

    if (alg == 1)
        CleanUpClassical();
    ChangeTree(tree);
    alg = newAlg;
    if (alg == 0)
    {
        blockDim = dim3(BLOCKXSIZE, BLOCKYSIZE);
    }
    else if (alg == 1)
    {
        blockDim = dim3(BLOCKXSIZERAYMARCH, BLOCKYSIZERAYMARCH);
        SetupClassical(tree);
    }
    else if (alg == 2)
    {
        blockDim = dim3(BLOCKXSIZERAYMARCH, BLOCKYSIZERAYMARCH);
    }
    gridDim = dim3((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
}

void Raycaster::MapFromCamera(Camera cam)
{
    cudaCamera.position = make_float3(cam.x, cam.y, cam.z);
    cudaCamera.forward = make_float3(cam.forward[0], cam.forward[1], cam.forward[2]);
    cudaCamera.right = make_float3(cam.right[0], cam.right[1], cam.right[2]);
    cudaCamera.up = make_float3(cam.up[0], cam.up[1], cam.up[2]);
    cudaCamera.fov = cam.fov;
}