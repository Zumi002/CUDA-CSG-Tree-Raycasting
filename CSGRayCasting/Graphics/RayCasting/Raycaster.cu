#include "Raycaster.cuh"
#include "Kernels/Kernels.cuh"

void Raycaster::ChangeTree(CSGTree& tree)
{
    CleanUpTree();
    gpuErrchk(cudaMalloc(&cudaTree.nodes, tree.nodes.size() * sizeof(CSGNode)));
    gpuErrchk(cudaMalloc(&cudaTree.primitives, tree.primitives.primitives.size() * sizeof(Primitive)));

    gpuErrchk(cudaMemcpy(cudaTree.nodes, tree.nodes.data(), tree.nodes.size() * sizeof(CSGNode), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(cudaTree.primitives, tree.primitives.primitives.data(), tree.primitives.primitives.size() * sizeof(Primitive), cudaMemcpyHostToDevice));
    allocedTree = true;
}

void Raycaster::ChangeSize(int newWidth, int newHeight)
{
    CleanUpTexture();
    width = newWidth;
    height = newHeight;
    blockDim = dim3(BLOCKXSIZE, BLOCKYSIZE);
    gridDim = dim3((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
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

    if (alg == 0)
        RaycastKernel << <gridDim, blockDim >> > (cudaCamera, cudaTree, devHits, width, height);
    else if (alg == 1)
        CalculateInterscetion << <gridDim, blockDim >> > (width, height, shapeCount, cudaTree, devParts, cudaCamera, devHits);
    cudaDeviceSynchronize();



    LightningKernel << <gridDim, blockDim >> > (cudaCamera, devHits, cudaTree.primitives, devPBO, light.getLightDir(), width, height);
    cudaDeviceSynchronize();

}

void Raycaster::CleanUpTree()
{
    if (allocedTree)
    {
        gpuErrchk(cudaFree(cudaTree.nodes));
        gpuErrchk(cudaFree(cudaTree.primitives));
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
    if (allocedTree)
    {
        Parts = (int*)malloc(tree.primitives.primitives.size() * 4 * sizeof(int));
        CreateParts(tree, Parts, 0);
        gpuErrchk(cudaMalloc(&devParts, tree.primitives.primitives.size() * 4 * sizeof(int)));
        gpuErrchk(cudaMemcpy(devParts, Parts, tree.primitives.primitives.size() * 4 * sizeof(int), cudaMemcpyHostToDevice));
        shapeCount = tree.primitives.primitives.size();
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
    if (alg == 1)
        SetupClassical(tree);

}

void Raycaster::MapFromCamera(Camera cam)
{
    cudaCamera.position = make_float3(cam.x, cam.y, cam.z);
    cudaCamera.forward = make_float3(cam.forward[0], cam.forward[1], cam.forward[2]);
    cudaCamera.right = make_float3(cam.right[0], cam.right[1], cam.right[2]);
    cudaCamera.up = make_float3(cam.up[0], cam.up[1], cam.up[2]);
    cudaCamera.fov = cam.fov;
}