#include "Raycaster.cuh"
#include "Kernels/Kernels.cuh"

Raycaster::Raycaster()
{
    int sharedMemPerBlock;
    cudaDeviceGetAttribute(&sharedMemPerBlock, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    gpuErrchk(cudaFuncSetAttribute(RaycastKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(CalculateInterscetionShared<32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(CalculateInterscetionShared<64>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(CalculateInterscetionShared<128>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(CalculateInterscetionShared<256>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(CalculateInterscetionShared<512>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(CalculateInterscetionShared<1024>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(CalculateInterscetionShared<2048>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(RaymarchingKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));

    gpuErrchk(cudaFuncSetAttribute(LightningKernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sharedMemPerBlock));
}

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
    shapeCount = tree.primitives.primitivePos.size();
}

void Raycaster::ChangeSize(int newWidth, int newHeight)
{
    CleanUpTexture();
    width = newWidth;
    height = newHeight;
    gpuErrchk(cudaMalloc(&devHits, width * height * sizeof(RayHit)));
    alloced = true;
    CalculateBlockSizes();
}

void Raycaster::ChangeSize(int newWidth, int newHeight, CSGTree& tree)
{
    ChangeSize(newWidth, newHeight);
    ChangeTree(tree);
}

void Raycaster::Raycast(float4* devPBO, Camera cam, DirectionalLight light)
{
    MapFromCamera(cam);
    cuProfilerStart();
    if (alg == 0)
        RaycastKernel << <gridDimSingle, blockDimSingle >> > (cudaCamera, cudaTree.nodes, devBvhNodes, cudaTree.primitivePos, cudaTree.primitiveParams, devHits, width, height);
    else if (alg == 1)
    {
        if (shapeCount <= 32)
        {
            CalculateInterscetionShared<32><< <gridDimClassic, blockDimClassic, 8192 >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
        }
        else if (shapeCount <= 64)
        {
            CalculateInterscetionShared<64> << <gridDimClassic, blockDimClassic, 8192 >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
        }
        else if (shapeCount <= 128)
        {
            CalculateInterscetionShared<128> << <gridDimClassic, blockDimClassic, 8192 >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
        }
        else if (shapeCount <= 256)
        {
            CalculateInterscetionShared<256> << <gridDimClassic, blockDimClassic, 8192 >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
        }
        else if (shapeCount <= 512)
        {
            CalculateInterscetionShared<512> << <gridDimClassic, blockDimClassic, 8192 >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
        }
        else if (shapeCount <= 1024)
        {
            CalculateInterscetionShared<1024> << <gridDimClassic, blockDimClassic, 8192 >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
        }
        else if (shapeCount <= 2048)
        {
            CalculateInterscetionShared<2048> << <gridDimClassic, blockDimClassic, 8192 >> > (width, height, shapeCount, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devParts, cudaCamera, devHits);
        }
    }
    else if (alg == 2)
    {
        if (nodeCount <= RAYMARCHSHAREDNODES)
        {
            RaymarchingKernelShared << <gridDimRayMarch, blockDimRayMarch >> > (cudaCamera, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devHits, nodeCount, width, height);
        }
        else
        {
            RaymarchingKernel << <gridDimRayMarch, blockDimRayMarch >> > (cudaCamera, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, devHits, nodeCount, width, height);
        }
    }

    if (collectStats)
    {
        cudaMemset(devStats, 0, 2 * sizeof(int));
        PrimitivePerPixelStatistic<<<gridDimRayMarch, blockDimRayMarch>>>(cudaCamera, cudaTree.nodes, cudaTree.primitivePos, cudaTree.primitiveParams, width, height, shapeCount, devStats);
        cudaMemcpy(stats, devStats, 2 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cuProfilerStop();

    LightningKernel << <gridDimLighting, blockDimLighting >> > (cudaCamera, devHits, cudaTree.primitiveColor, devPBO, light.getLightDir(), width, height);
    cudaDeviceSynchronize();
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
        allocedClassicalAdds = true;
    }
}

void Raycaster::CleanUp()
{
    CleanUpTree();
    CleanUpTexture();
    CleanUpClassical();
    CleanUpSingleHit();

    if (collectStats)
    {
        gpuErrchk(cudaFree(devStats));
        delete[] stats;
    }
}

void Raycaster::ChangeAlg(CSGTree& tree, int newAlg)
{
    if (alg == 0)
    {
        CleanUpSingleHit();
    }
    else if (alg == 1)
    {
        CleanUpClassical();
    }

    ChangeTree(tree);
    alg = newAlg;

    if (alg == 0)
    {
        SetupSingleHit(tree);
    }
    else if (alg == 1)
    {
        SetupClassical(tree);
    }
    CalculateBlockSizes();
}



void Raycaster::MapFromCamera(Camera cam)
{
    cudaCamera.position = make_float3(cam.x, cam.y, cam.z);
    cudaCamera.forward = make_float3(cam.forward[0], cam.forward[1], cam.forward[2]);
    cudaCamera.right = make_float3(cam.right[0], cam.right[1], cam.right[2]);
    cudaCamera.up = make_float3(cam.up[0], cam.up[1], cam.up[2]);
    cudaCamera.fov = cam.fov;
}

void Raycaster::CalculateBlockSizes()
{
    blockDimSingle = dim3(BLOCKXSIZE, BLOCKYSIZE);
    gridDimSingle = dim3((width + blockDimSingle.x - 1) / blockDimSingle.x, (height + blockDimSingle.y - 1) / blockDimSingle.y);
    blockDimClassic = dim3(BLOCKXSIZECLASSIC, BLOCKYSIZECLASSIC);
    gridDimClassic = dim3((width + blockDimClassic.x - 1) / blockDimClassic.x, (height + blockDimClassic.y - 1) / blockDimClassic.y);
    blockDimRayMarch = dim3(BLOCKXSIZERAYMARCH, BLOCKYSIZERAYMARCH);
    gridDimRayMarch = dim3((width + blockDimRayMarch.x - 1) / blockDimRayMarch.x, (height + blockDimRayMarch.y - 1) / blockDimRayMarch.y);
    blockDimLighting = dim3(BLOCKXSIZELIGHTING, BLOCKYSIZELIGHTING);
    gridDimLighting = dim3((width + blockDimLighting.x - 1) / blockDimLighting.x, (height + blockDimLighting.y - 1) / blockDimLighting.y);
}

void Raycaster::SetupSingleHit(CSGTree& tree)
{
    if (allocedTree && tree.nodes.size())
    {
        std::vector<BVHNode> bvhNodes = tree.ConstructBVH();
        gpuErrchk(cudaMalloc(&devBvhNodes, bvhNodes.size()*sizeof(BVHNode)));
        gpuErrchk(cudaMemcpy(devBvhNodes, bvhNodes.data(), bvhNodes.size() * sizeof(BVHNode), cudaMemcpyHostToDevice));
        allocedSingleHitAdds = true;
    }
}

void Raycaster::CleanUpSingleHit()
{
    if (allocedSingleHitAdds)
    {
        gpuErrchk(cudaFree(devBvhNodes));
        allocedSingleHitAdds = false;
    }
}

void Raycaster::CollectStats()
{
    collectStats = true;
    gpuErrchk(cudaMalloc(&devStats, 2 * sizeof(int)));
    stats = new int[2];
}