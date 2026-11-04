#pragma once
#include "point3d.hpp"

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <tuple>

namespace manager_v3
{
/**
 * @brief      Gets the maximum dynamic shared memory for the current device.
 * @return     The maximum dynamic shared memory.
 */
inline int getSharedMemPerBlock()
{
    int max_dynamic_shared_mem = 0;
    cudaDeviceGetAttribute(&max_dynamic_shared_mem,
                           cudaDevAttrMaxSharedMemoryPerBlock, 0);
    return max_dynamic_shared_mem;
}

/**
 * @brief      Gets the number of multiprocessors for the current device.
 * @return     The number of multiprocessors.
 */
inline int getMultiProcessorCount()
{
    int nb_processors = 0;
    cudaDeviceGetAttribute(&nb_processors, cudaDevAttrMultiProcessorCount, 0);
    return nb_processors;
}

/**
 * @brief      Gets the maximum dynamic shared memory for the current device.
 * @return     The maximum dynamic shared memory.
 */
inline int getMaxDynamicSM()
{
    int max_dynamic_shared_mem = 0;
    cudaDeviceGetAttribute(&max_dynamic_shared_mem,
                           cudaDevAttrMaxSharedMemoryPerBlock, 0);
    return max_dynamic_shared_mem;
}

inline int getMaxThreadPerBlock(){
    int max_thread_per_block = 0;
    cudaDeviceGetAttribute(&max_thread_per_block,
                           cudaDevAttrMaxThreadsPerBlock, 0);
    return max_thread_per_block;
}

// Get gpu type
inline std::string getGpuName(){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    return deviceProp.name;
}

/**
 * @brief      Compute the configuration for kernel launch. It looks for the
 * optimal number of cells that can be handled by a block and it defined
 * the grid dimension to handle all the cells of the simulation. Then, it
 * defined the number of thread per block to handle all the particles
 * in the number of cells.
 *
 * For example suppose we have \p simBoxGrid = (15, 15, 15),
 * \p maxParticlesPerCell = 60 and \p memPerCell = 7000
 * Then only one cell will be handle by a block and therefore it will return :
 * A grid dimension of (15, 15, 15) as each cells is handled by one block
 * and each block will have 60 threads which corresponds to the maximum number
 * of particles per cells (equivalent to the maximum number of particles per
 * block as we consider only one cell per block).
 *
 * @warning    It assumes that each cells won't have more than
 * \p maxParticlesPerCell per cells.
 *
 *
 * @return     A tuple that contains the grid dimension and the number of thread
 * per block
 */
inline auto getSharedmemConf(const int3 simBoxGrid,
                             const int maxParticlesPerCell,
                             const int memPerCell)
{
    const int nbrTotalCells = simBoxGrid.x * simBoxGrid.y * simBoxGrid.z;

    // Compute the number of threads with min and max bound
    // Weird but T600 cannot have more 512
    const int MaxNbThreads = (getGpuName().find("T600") != std::string::npos ? 512 : getMaxThreadPerBlock());
    const int MinNbThreads = 32;

    // The maximum number of cells a block can manage due to the total number of
    // cells constraints.
    const int MaxNbCellsPerBlock1Dim = std::max(1, nbrTotalCells / getMultiProcessorCount());
    // The maximum number of cells a block can manage due to shared memory
    // constraints (including ghosts).
    const int MaxNbCellsPerMem1Dim = getSharedMemPerBlock() / memPerCell;
    if(MaxNbCellsPerMem1Dim < 3){ // We need at least 3 cells for a xpencil
        return std::make_tuple(false, int3{0, 0, 0}, 0);
    }
    // The width of the cube that respect the constraints
    // it includes the neighbors
    const int MaxNbCellsPerBlockThreadLimit = MaxNbThreads/maxParticlesPerCell;
    int MaxCellPerBlockX = std::min({int(MaxNbCellsPerMem1Dim), 
                                    MaxNbCellsPerBlock1Dim+2,
                                    MaxNbCellsPerBlockThreadLimit+2});
    assert(MaxCellPerBlockX >= 3);
    const int MaxNbCellsXY = std::max(1, int(sqrt(MaxNbCellsPerBlockThreadLimit/(MaxCellPerBlockX-2))));
    int MaxCellPerBlockY = MaxNbCellsXY;
    int MaxCellPerBlockZ = MaxNbCellsXY;

    const int HardLimitNbCells = std::min(MaxNbCellsPerBlockThreadLimit, MaxNbCellsPerBlock1Dim);

    // We will see if we can increase the size in one dimension while respecting
    // the constraints
    if (((MaxCellPerBlockZ) + 1) * (MaxCellPerBlockY) * (MaxCellPerBlockX-2) <= HardLimitNbCells)
    {
        MaxCellPerBlockZ += 1;

        if ((MaxCellPerBlockZ) * ((MaxCellPerBlockY) + 1) * (MaxCellPerBlockX-2) <= HardLimitNbCells)
        {
            MaxCellPerBlockY += 1;
        }
        else if (((MaxCellPerBlockZ) + 1) * (MaxCellPerBlockY) * (MaxCellPerBlockX-2) <= HardLimitNbCells)
        {
            MaxCellPerBlockZ += 1;
        }
    }
    bool hasImproved = true;
    while(hasImproved){
        hasImproved = false;

        const double currentRatio = double(MaxCellPerBlockZ * MaxCellPerBlockY * (MaxCellPerBlockX-2))
                                    / double((MaxCellPerBlockZ+2) * (MaxCellPerBlockY+2) * MaxCellPerBlockX);
        if(MaxCellPerBlockZ <= MaxCellPerBlockY){
            const int newZ = MaxCellPerBlockZ + 1;
            const int newX = HardLimitNbCells/(newZ * MaxCellPerBlockY) + 2;
            if(newX >= 3){
                const double newRatio = double(newZ * MaxCellPerBlockY * (newX-2))
                                        / double((newZ+2) * (MaxCellPerBlockY+2) * newX);
                if(newRatio > currentRatio){
                    MaxCellPerBlockZ = newZ;
                    MaxCellPerBlockX = newX;
                    hasImproved = true;
                }
            }
        }
        else{
            const int newY = MaxCellPerBlockY + 1;
            const int newX = HardLimitNbCells/(MaxCellPerBlockZ * newY) + 2;
            if(newX >= 3){
                const double newRatio = double(MaxCellPerBlockZ * newY * (newX-2))
                                        / double((MaxCellPerBlockZ+2) * (newY+2) * newX);
                if(newRatio > currentRatio){
                    MaxCellPerBlockY = newY;
                    MaxCellPerBlockX = newX;
                    hasImproved = true;
                }
            }
        }
    }
    assert(MaxCellPerBlockX >= 3 && MaxCellPerBlockY >= 1 &&
           MaxCellPerBlockZ >= 1);

    // We have included the neighbors cells for X, and now we remove them
    MaxCellPerBlockX = MaxCellPerBlockX - 2;

    // We compute the number of blocks
    const int NbBlocksX =
        std::max(1, (simBoxGrid.x + MaxCellPerBlockX - 1) / MaxCellPerBlockX);
    const int NbBlocksY =
        std::max(1, (simBoxGrid.y + MaxCellPerBlockY - 1) / MaxCellPerBlockY);
    const int NbBlocksZ =
        std::max(1, (simBoxGrid.z + MaxCellPerBlockZ - 1) / MaxCellPerBlockZ);
    // Store it in a int3
    const int3 dimGrid{NbBlocksX, NbBlocksY, NbBlocksZ};


    // max number of particles per block (max number of particles per cell time
    // the number of cell per block in 3D)
    const int maxNbrParticlesPerBlock = MaxCellPerBlockX * MaxCellPerBlockY *
                                        MaxCellPerBlockZ * maxParticlesPerCell;
    // we make sure that the number of particles per block is greater than the
    // minimum number of threads
    const int nbrThreadPerBlock =
        std::max(MinNbThreads, maxNbrParticlesPerBlock);
    assert(nbrThreadPerBlock <= MaxNbThreads);
    // Ideally the number of threads is the expected number of particles for the
    // number of cells per block, but we make sure it is between MinNbThreads
    // and MaxNbThreads.
    const int NbThreads = std::min(MaxNbThreads, nbrThreadPerBlock);
    return std::make_tuple(true, dimGrid, NbThreads);
}

void kernelCallerCudaMemcpy3FToF3(float3 *dest, const float *srcX,
                                  const float *srcY, const float *srcZ,
                                  size_t count);

/**
 * @class      SphInteractionManager
 *
 * @brief      This class handle the way how particles interact between them.
 * Over a question of performance particles only interact with there neighbors.
 * The first approach is to used a grid and when computing the data for one
 * particles, we check its cell and the cells around him (in 3D its 3^3 = 27
 * cells). To improve the performance of this approach we propose a more
 * sophisticated version of this. Our GPU are slow to read and write memory,
 * therefore we tackle this problem by putting most of the read and write in
 * shared memory. To do that, we will focus our work on grid and not on
 * particles. We need to compute some array that will move the way the grid is
 * handle from global to local for shared memory. The \ref init function is
 * the core of this class in will compute the index in a way that each block of
 * GPU will focus its work on some calculation of a few cells and their
 * neighbors. More concretely, it defined all the index array that will be used
 * to identify the cells in \ref computeInteraction.
 */
template <class RealType, class IndexType> class SphInteractionManager
{
    unsigned char *const bufferOfShared;
    unsigned char *ptrToBufferOfSharedMem;

    __forceinline__ __device__ IndexType iabs(const IndexType inValue)
    {
        return inValue < 0 ? -inValue : inValue;
    }

    public:
    __forceinline__ __device__ explicit SphInteractionManager(
        unsigned char *inBfufferOfShared)
        : bufferOfShared(inBfufferOfShared)
    {
        ptrToBufferOfSharedMem = &bufferOfShared[0];
    }

    template <class ObjType>
    __forceinline__ __device__ ObjType *getSmBlock(const IndexType inNbElements)
    {
        ptrToBufferOfSharedMem += (sizeof(ObjType) -
                            (size_t(ptrToBufferOfSharedMem) % sizeof(ObjType))) %
                           sizeof(ObjType);
        ObjType *blockAddress =
            reinterpret_cast<ObjType *>(ptrToBufferOfSharedMem);
        ptrToBufferOfSharedMem += sizeof(ObjType) * inNbElements;
        return blockAddress;
    }

    /**
     * @brief      Initialize the
     *
     * @param[in]  positionsParticlesX  The positions of particles on x axis
     * @param[in]  positionsParticlesY  The positions of particles on y axis
     * @param[in]  positionsParticlesZ  The positions of particles on z axis
     * @param[in]  gridCellsCounts      The global offset to determine the
     * number of particles in each cells.
     */    
    template <class FuncCopy, class FuncCopyRegisters, class FuncInitType, class FuncInteractType, class FuncReleaseType>
    __forceinline__ __device__ void compute(
        const RealType *const __restrict__ positionsParticlesX,
        const RealType *const __restrict__ positionsParticlesY,
        const RealType *const __restrict__ positionsParticlesZ,
        const IndexType *const __restrict__ gridCellsCounts,
        const Point3D<IndexType> nbCells, const IndexType inMaxPartsPerCell,
        FuncCopy &&inFuncCopy, FuncCopyRegisters &&inFuncCopyRegisters, FuncInitType &&FuncInit,
        FuncInteractType &&FuncInteract, FuncReleaseType &&FuncRelease)
    {
        // The size of the subgrid of cells we have to manage
        const IndexType nbCellsBlockX = (nbCells.x + gridDim.x - 1) / gridDim.x;
        const IndexType nbCellsBlockY = (nbCells.y + gridDim.y - 1) / gridDim.y;
        const IndexType nbCellsBlockZ = (nbCells.z + gridDim.z - 1) / gridDim.z;
        // In each dim, we compute the indices of the first and last cell
        // we make sure to always be in the grid (eg. the cells on the corner
        // will be there own limits).
        // Extra step is the width of extra cells we take
        constexpr IndexType extraStep = 1;
        const IndexType firstCellX = blockIdx.x * nbCellsBlockX;
        const IndexType lastCellX = min(nbCells.x, firstCellX + nbCellsBlockX);
        const IndexType nbCellsX = lastCellX - firstCellX;

        const IndexType firstCellY = blockIdx.y * nbCellsBlockY;
        const IndexType lastCellY = min(nbCells.y, firstCellY + nbCellsBlockY);
        const IndexType nbCellsY = lastCellY - firstCellY;

        const IndexType firstCellZ = blockIdx.z * nbCellsBlockZ;
        const IndexType lastCellZ = min(nbCells.z, firstCellZ + nbCellsBlockZ);
        const IndexType nbCellsZ = lastCellZ - firstCellZ;
        const IndexType totalNbCells = nbCellsX * nbCellsY * nbCellsZ;
        // The number of threads must be greater than the number of particles
        assert(totalNbCells * inMaxPartsPerCell <= blockDim.x);

        const IndexType firstCellXWithGhost = max(0, firstCellX - extraStep);
        const IndexType lastCellXWithGhost = min(nbCells.x, lastCellX + extraStep);
        const IndexType nbCellsXWithGhost = lastCellXWithGhost - firstCellXWithGhost;

        const IndexType firstCellYWithGhost = max(0, firstCellY - extraStep);
        const IndexType lastCellYWithGhost = min(nbCells.y, lastCellY + extraStep);
        const IndexType nbCellsYWithGhost = lastCellYWithGhost - firstCellYWithGhost;

        const IndexType firstCellZWithGhost = max(0, firstCellZ - extraStep);
        const IndexType lastCellZWithGhost = min(nbCells.z, lastCellZ + extraStep);
        const IndexType nbCellsZWithGhost = lastCellZWithGhost - firstCellZWithGhost;
        assert(nbCellsXWithGhost > 0 && nbCellsYWithGhost > 0 &&
               nbCellsZWithGhost > 0);

        // The total number of cells we will need to load data from
        const IndexType totalNbCellsWithGohst = nbCellsXWithGhost * nbCellsYWithGhost * nbCellsZWithGhost;
        // Since we copy the offset from global memory, it is in the X dim, and so we keep
        // one additional X values (so for 1 * nbCellsYWithGhost * nbCellsZWithGhost)
        const IndexType totalNbCellsWithGostP1 = totalNbCellsWithGohst + nbCellsYWithGhost * nbCellsZWithGhost;
        const IndexType nbCellsXP1 = nbCellsXWithGhost + 1;
        
        IndexType* cellOffsetP1 = getSmBlock<IndexType>(totalNbCellsWithGostP1);
        for(IndexType idxCell = threadIdx.x ; idxCell < totalNbCellsWithGostP1 ; idxCell += blockDim.x){
            // Retreive the cell index in the local grid
            const IndexType idxXP1 = idxCell % nbCellsXP1;
            const IndexType idxYP1 = (idxCell / nbCellsXP1) % nbCellsYWithGhost;
            const IndexType idxZP1 = idxCell / (nbCellsXP1 * nbCellsYWithGhost);
            // Compute the cell index in the global grid
            const IndexType idxGlobalX = idxXP1 + firstCellXWithGhost;
            const IndexType idxGlobalY = idxYP1 + firstCellYWithGhost;
            const IndexType idxGlobalZ = idxZP1 + firstCellZWithGhost;
            // Unique index of the cell
            const IndexType idxGlobal = (idxGlobalZ * nbCells.y + idxGlobalY) * nbCells.x + idxGlobalX;
            // Copy the offset
            cellOffsetP1[idxCell] = gridCellsCounts[idxGlobal];
        }

        __syncthreads();

        // Will be filled if possible
        IndexType myCellXWithGhost;
        IndexType myCellYWithGhost;
        IndexType myCellZWithGhost;

        IndexType nbParticulesInMyCell;
        IndexType myParticleGlobalOffset;
        Point3D<RealType> xI;

        // Each thread finds it corresponding cell
        const IndexType MaxPartPerXPencil = inMaxPartsPerCell * nbCellsXWithGhost;
        const IndexType myXpencilIdx = threadIdx.x / MaxPartPerXPencil;
        bool threadCompute = false;
        if(myXpencilIdx < nbCellsY*nbCellsZ){
            const IndexType myPartIdxInXpencil = threadIdx.x % MaxPartPerXPencil;
            const IndexType xPencilY = myXpencilIdx % nbCellsY;
            const IndexType xPencilZ = myXpencilIdx / nbCellsY;

            myCellYWithGhost = (firstCellYWithGhost < firstCellY ? xPencilY + 1 : xPencilY);
            myCellZWithGhost = (firstCellZWithGhost < firstCellZ ? xPencilZ + 1 : xPencilZ);

            const IndexType firstCellOfXpencilP1 = (myCellZWithGhost * nbCellsYWithGhost + myCellYWithGhost) * nbCellsXP1
                                                    + (firstCellXWithGhost < firstCellX ? 1 : 0);
            const IndexType lastCellOfXpencilP1 = firstCellOfXpencilP1 + nbCellsX;
            const IndexType nbParticlesInMyXpencil = cellOffsetP1[lastCellOfXpencilP1] - cellOffsetP1[firstCellOfXpencilP1];

            assert(lastCellOfXpencilP1-firstCellOfXpencilP1 <= nbCellsXWithGhost);
            assert(nbParticlesInMyXpencil <= MaxPartPerXPencil);

            if(myPartIdxInXpencil < nbParticlesInMyXpencil){
                threadCompute = true;

                myParticleGlobalOffset = cellOffsetP1[firstCellOfXpencilP1] + myPartIdxInXpencil;
                xI = Point3D<RealType>{positionsParticlesX[myParticleGlobalOffset],
                    positionsParticlesY[myParticleGlobalOffset],
                    positionsParticlesZ[myParticleGlobalOffset]};
                FuncInit(myParticleGlobalOffset);

                IndexType myCellIdx;
                {// Binary search to find the right cell
                    IndexType iterFirstCellX = firstCellOfXpencilP1;
                    IndexType iterLastCellX = lastCellOfXpencilP1;
                    while(iterFirstCellX < iterLastCellX){
                        myCellIdx = (iterFirstCellX + iterLastCellX) / 2;
                        const IndexType firstCellXP1 = myCellIdx;
                        const IndexType lastCellXP1 = firstCellXP1 + 1;
                        const IndexType startPartIdx = cellOffsetP1[firstCellXP1];
                        const IndexType endPartIdx = cellOffsetP1[lastCellXP1];
                        if(startPartIdx <= myParticleGlobalOffset && myParticleGlobalOffset < endPartIdx){
                            break;
                        }
                        else if(myParticleGlobalOffset < startPartIdx){
                            iterLastCellX = myCellIdx;
                        }
                        else{
                            iterFirstCellX = myCellIdx + 1;
                        }
                    }
                }

                assert(firstCellOfXpencilP1 <= myCellIdx);
                assert(myCellIdx < lastCellOfXpencilP1);
                assert(cellOffsetP1[myCellIdx] <= myParticleGlobalOffset && myParticleGlobalOffset < cellOffsetP1[myCellIdx+1]);

                myCellXWithGhost = myCellIdx - firstCellOfXpencilP1 + (firstCellXWithGhost < firstCellX ? 1 : 0);

                assert(0 < myCellXWithGhost || firstCellXWithGhost == firstCellX);
                assert(myCellXWithGhost < nbCellsXWithGhost);
            }
        }

        // The maximum number of particles we can have in the xpencil
        const IndexType myNbParticlePerXPencil = nbCellsXWithGhost * inMaxPartsPerCell;
        RealType* srcX = getSmBlock<RealType>(myNbParticlePerXPencil);
        RealType* srcY = getSmBlock<RealType>(myNbParticlePerXPencil);
        RealType* srcZ = getSmBlock<RealType>(myNbParticlePerXPencil);
      
        // loop on all neighbors cells, including those that some of the threads
        // do not use at that iterations
        for (IndexType idxNeighZ = firstCellZWithGhost;
                idxNeighZ < lastCellZWithGhost; ++idxNeighZ)
        {
            for (IndexType idxNeighY = firstCellYWithGhost;
                    idxNeighY < lastCellYWithGhost; ++idxNeighY)
            {
                // The index of the xpencil in the sub box
                const IndexType idxNeighZP1 = idxNeighZ-firstCellZWithGhost;
                const IndexType idxNeighYP1 = idxNeighY-firstCellYWithGhost;
                const IndexType firstCellXPencilP1 = (idxNeighZP1 * nbCellsYWithGhost + idxNeighYP1) * nbCellsXP1;
                const IndexType limiteCellXPencilP1 = firstCellXPencilP1 + nbCellsXP1 - 1;

                // Full XPencil should be copied from global memory (we are on the border of the subbox)
                if(idxNeighZ < firstCellZ || lastCellZ < idxNeighZ
                     || idxNeighY < firstCellY || lastCellY < idxNeighY){
                    const IndexType startPartIdx = cellOffsetP1[firstCellXPencilP1];
                    const IndexType endPartIdx = cellOffsetP1[limiteCellXPencilP1];

                    assert(endPartIdx - startPartIdx <= myNbParticlePerXPencil);

                    for(int idxCopy = threadIdx.x ; idxCopy < endPartIdx - startPartIdx ; idxCopy += blockDim.x){
                        srcX[idxCopy] = positionsParticlesX[startPartIdx + idxCopy];
                        srcY[idxCopy] = positionsParticlesY[startPartIdx + idxCopy];
                        srcZ[idxCopy] = positionsParticlesZ[startPartIdx + idxCopy];
                        inFuncCopy(idxCopy, startPartIdx + idxCopy);
                    }
                }
                else{
                    // Number of cells before the xpencil in the local offset array

                    // We only need to copy the first and last cell of the xpencil (if they exist)
                    // the rest of the values will be copied form the registers
                    // Copy first cell ? (their is a ghost cell on the left)
                    if(firstCellXWithGhost < firstCellX){
                        const IndexType lastCellXPencilP1 = firstCellXPencilP1 + 1;

                        const IndexType startPartIdx = cellOffsetP1[firstCellXPencilP1];
                        const IndexType endPartIdx = cellOffsetP1[lastCellXPencilP1];

                        assert(endPartIdx - startPartIdx <= inMaxPartsPerCell);

                        for(int idxCopy = threadIdx.x ; idxCopy < endPartIdx - startPartIdx ; idxCopy += blockDim.x){
                            srcX[idxCopy] = positionsParticlesX[startPartIdx + idxCopy];
                            srcY[idxCopy] = positionsParticlesY[startPartIdx + idxCopy];
                            srcZ[idxCopy] = positionsParticlesZ[startPartIdx + idxCopy];
                            inFuncCopy(idxCopy, startPartIdx + idxCopy);
                        }
                    }
                    // Copy last cell ? (their is a ghost cell on the right)
                    if(lastCellX < lastCellXWithGhost){
                        const IndexType offsetPartIdx = cellOffsetP1[limiteCellXPencilP1-1] - cellOffsetP1[firstCellXPencilP1];
                        const IndexType startPartIdx  = cellOffsetP1[limiteCellXPencilP1-1];
                        const IndexType endPartIdx    = cellOffsetP1[limiteCellXPencilP1];

                        assert(offsetPartIdx + endPartIdx - startPartIdx <= myNbParticlePerXPencil);
                        assert(endPartIdx - startPartIdx <= inMaxPartsPerCell);

                        for(int idxCopy = threadIdx.x ; idxCopy < endPartIdx - startPartIdx ; idxCopy += blockDim.x){
                            srcX[idxCopy+offsetPartIdx] = positionsParticlesX[startPartIdx + idxCopy];
                            srcY[idxCopy+offsetPartIdx] = positionsParticlesY[startPartIdx + idxCopy];
                            srcZ[idxCopy+offsetPartIdx] = positionsParticlesZ[startPartIdx + idxCopy];
                            inFuncCopy(idxCopy+offsetPartIdx, startPartIdx + idxCopy);
                        }
                    }
                    // Same xpencil and has  particle in registers
                    if(idxNeighZP1 == myCellZWithGhost && idxNeighYP1 == myCellYWithGhost && threadCompute){
                        const IndexType offsetPartIdx = cellOffsetP1[firstCellXPencilP1];
                        assert(myParticleGlobalOffset-offsetPartIdx < myNbParticlePerXPencil);
                        srcX[myParticleGlobalOffset-offsetPartIdx] = xI.x;
                        srcY[myParticleGlobalOffset-offsetPartIdx] = xI.y;
                        srcZ[myParticleGlobalOffset-offsetPartIdx] = xI.z;
                        inFuncCopyRegisters(myParticleGlobalOffset-offsetPartIdx, myParticleGlobalOffset);
                    }
                }

                __syncthreads();

                // Loop on all particles in the xpencil if have to compute with them
                if(threadCompute && iabs(idxNeighZP1-myCellZWithGhost) <= 1 && iabs(idxNeighYP1-myCellYWithGhost) <= 1){
                    // Find particle interval
                    const IndexType firstCellX = max(0, myCellXWithGhost - 1);
                    const IndexType lastCellX = min(nbCellsXWithGhost, myCellXWithGhost + 2);

                    const IndexType firstCellComputeXPencilP1 = (idxNeighZP1 * nbCellsYWithGhost + idxNeighYP1) * nbCellsXP1 + firstCellX;
                    const IndexType lastCellComputeXPencilP1 = (idxNeighZP1 * nbCellsYWithGhost + idxNeighYP1) * nbCellsXP1 + lastCellX;
                    
                    const IndexType offsetPartIdx = cellOffsetP1[firstCellXPencilP1];
                    const IndexType firstParticleIdx = cellOffsetP1[firstCellComputeXPencilP1]-offsetPartIdx;
                    const IndexType lastParticleIdx = cellOffsetP1[lastCellComputeXPencilP1]-offsetPartIdx;

                    assert(lastParticleIdx - firstParticleIdx <= 3*inMaxPartsPerCell);

                    for(IndexType idxPart = firstParticleIdx ; idxPart < lastParticleIdx ; ++idxPart){
                        if(idxPart+offsetPartIdx != myParticleGlobalOffset){
                            const Point3D<RealType> xJ{
                                srcX[idxPart],
                                srcY[idxPart],
                                srcZ[idxPart]};
                            FuncInteract(xI, xJ, idxPart, idxPart+offsetPartIdx);
                        }
                    }
                }

                __syncthreads();
            }
        }

        if(threadCompute){
            FuncRelease(myParticleGlobalOffset, xI);
        }
    }

    /**
     * @brief      Determined the size of shared memory for one block from the
     * max number of particles in one cell.
     *
     * @param[in]  maxParticlesPerCell  The theoretical maximum number of
     * particle per cell
     *
     * @return     The size of byte of shared memory for one block. This can be
     * used for kernel launching.
     */
    __host__ static IndexType sharedMemSize(
        const Point3D<IndexType> nbrCells,
        const Point3D<IndexType> expectedGridDim,
        const IndexType maxPartPerCell,
        const IndexType additionalSizeSMPerParticle = IndexType(0))
    {
        static_assert(std::is_signed<IndexType>::value, "Must be signed");

        const IndexType widthCellX =
            (nbrCells.x + expectedGridDim.x - 1) / expectedGridDim.x + 2;
        const IndexType widthCellY =
            (nbrCells.y + expectedGridDim.y - 1) / expectedGridDim.y + 2;
        const IndexType widthCellZ =
            (nbrCells.z + expectedGridDim.z - 1) / expectedGridDim.z + 2;

        const IndexType sizeOfIndexType = sizeof(IndexType);
        const IndexType sizeOfRealType = sizeof(RealType);

        const IndexType totalNbCellsInBlock =
            widthCellX * widthCellY * widthCellZ;
        const IndexType totalNbParticles = widthCellX * maxPartPerCell;
        const IndexType nbExtraOffsetValues = widthCellY * widthCellZ;

        const IndexType memorySizeForFirstTerm =
            sizeOfIndexType * (totalNbCellsInBlock + nbExtraOffsetValues);

        const IndexType memorySizeForSecondTerm =
            std::max(IndexType(0),
                     IndexType(sizeOfRealType - sizeOfIndexType)) +
            sizeOfRealType * totalNbParticles *
                (3 + additionalSizeSMPerParticle);

        const IndexType smNeededSize =
            memorySizeForFirstTerm + memorySizeForSecondTerm;

        assert(smNeededSize <= getSharedMemPerBlock());
        return smNeededSize;
    }

    /**
     * @brief      Determined the size of shared memory for one cell from the
     * max number of particles in one cell.
     *
     * @param[in]  maxParticlesPerCell  The theoretical maximum number of
     * particle per cell
     *
     * @return     The size of byte of shared memory for one cell.
     */
    __host__ static IndexType sharedMemSizeOneCell(
        const IndexType maxParticlesPerCell,
        const IndexType additionalSizeSMPerParticle = IndexType(0))
    {
        const IndexType totalCellsInBlock = 1;
        const IndexType numCellsLeftSideBlock = 1;
        // Calculate the size of shared memory required for one cell
        const IndexType sizeOfIndexType = sizeof(IndexType);
        const IndexType sizeOfRealType = sizeof(RealType);

        const IndexType memorySizeForFirstTerm =
            sizeOfIndexType * (totalCellsInBlock + 1);

        const IndexType memorySizeForSecondTerm =
            std::max(IndexType(0),
                     IndexType(sizeOfRealType - sizeOfIndexType)) +
            sizeOfRealType * maxParticlesPerCell *
                (totalCellsInBlock * 3 + additionalSizeSMPerParticle);

        // eg. for maxParticlesPerCell = 60, we get : 976
        return memorySizeForFirstTerm + memorySizeForSecondTerm;
    }

    /**
     * @brief      Compute the kernel launch configuration deduce from the
     * number of cells in each dimension and by assuming the cells can contains
     * only a limited number of particles (see maxNbPartPerCell).
     *
     * @param[in]  nbrCells  The number of cells in each dimension
     *
     * @param[in]  additionalSizeSMPerParticle     SphInteractionManager by
     * default initialize shared memory to contains some array specific to grid
     * computation and the position of particles (3 array of float), but some
     * kernels used more array for shared memory. For example, viscosity
     * computation in sph used velocity (3 array of float) and density (1 array
     * of float) therefore for each particles needs 4*sizeof(float) additional
     * byte, which can be pass to this parameter.
     *
     * @return     A tuple containing :
     * - the grid dimension (as a int3, use
     * \ref int3ToDim3 function to convert them back to dim3)
     * - the block dimension
     * - the shared memory size
     */
    __host__ static auto getKernelConf(
        const Point3D<IndexType> nbrCells, const IndexType maxParticlesPerCell,
        const IndexType additionalSizeSMPerParticle = 0)
    {
        const IndexType smSizePerCell = sharedMemSizeOneCell(
            maxParticlesPerCell, additionalSizeSMPerParticle);
        auto [enable, dimGrid, dimBlock] =
            getSharedmemConf(nbrCells, maxParticlesPerCell, smSizePerCell);
        if(enable == false){
            return std::make_tuple(false, int3{0, 0, 0}, 0, 0);
        }

        const IndexType smSize =
            sharedMemSize(nbrCells, dimGrid, maxParticlesPerCell,
                          additionalSizeSMPerParticle);

        return std::make_tuple(true, dimGrid, dimBlock, smSize);
    }
};


} 