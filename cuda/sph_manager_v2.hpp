#pragma once
#include "point3d.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <tuple>

namespace manager_v2
{

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
inline int getSharedMemPerBlock()
{
    int max_dynamic_shared_mem = 0;
    cudaDeviceGetAttribute(&max_dynamic_shared_mem,
                           cudaDevAttrMaxSharedMemoryPerBlock, 0);
    return max_dynamic_shared_mem;
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
                             const int memPerCell,
                             const int divSM = 3)
{
    // The number of XPencils (x rows)
    const int nbXPencils = simBoxGrid.y * simBoxGrid.z;
    // The maximum number of cells in the x direction
    const int maxXPencilLength = simBoxGrid.x;

    // The maximum number of cells a block can manage due to the total number of
    // cells constraints.
    // Consider getMultiProcessorCount() = 40, the number of x pencils is 10,
    // then we have 4 sub-pencil per pencil (so 4 blocks per pencil).
    const int BoundBlockPerXPencil =
        std::max(1, getMultiProcessorCount() / nbXPencils);
    // Then we find the number of cells per block
    const int MaxNbCellsPerBlock =
        std::max(1, maxXPencilLength / BoundBlockPerXPencil);
    // The maximum number of cells a block can manage due to shared memory
    // constraints.
    const int TMP_MAX_SM = getSharedMemPerBlock()/divSM;//14 * 1024;
    const int MaxCellInSharedMem = TMP_MAX_SM / memPerCell;
    // At least one cell and two ghost cells must fit in memory
    if(MaxCellInSharedMem < 3){
        return std::make_tuple(false, int3{0, 0, 0}, 0);
    }
    assert(MaxCellInSharedMem >= 3);
    // The max cells in shared memory without ghost cells
    const int MaxCellPerBlock = (MaxCellInSharedMem - 2);
    // The max cells considering that we should have one thread per particle
    const int MaxNbThreads = 1024;
    const int MaxCellPerBlockMaxThread = (MaxNbThreads/maxParticlesPerCell);
    if(MaxCellPerBlockMaxThread < 3){
        return std::make_tuple(false, int3{0, 0, 0}, 0);
    }
    const int MaxCellPerBlockMaxThreadNoGohst = (MaxCellPerBlockMaxThread-2);
    // We find the real number of cells that will be handle by one block (either
    // limited by the shared memory, or by the parallelism)
    const int NbCellsPerBlock = std::min(MaxCellPerBlock, std::min(MaxNbCellsPerBlock, MaxCellPerBlockMaxThreadNoGohst));

    // We compute the number of blocks
    const int NbBlocksX =
        std::max(1, (simBoxGrid.x + NbCellsPerBlock - 1) / NbCellsPerBlock);
    const int NbBlocksY = simBoxGrid.y;
    const int NbBlocksZ = simBoxGrid.z;
    // Store it in a int3
    const int3 dimGrid{NbBlocksX, NbBlocksY, NbBlocksZ};

    // Compute the number of threads with min and max bound
    const int MinNbThreads = 32;

    // max number of particles per block (max number of particles per cell time
    // the number of cell per block in 3D)
    const int maxNbrParticlesPerBlock =
        (NbCellsPerBlock + 2) * maxParticlesPerCell;
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
    unsigned char*const bufferOfShared;
    unsigned char* sharedMemBuffer;
    const IndexType maxPartPerCell;
    const Point3D<IndexType> nbCells;
    const IndexType nbCellsBlockCompute;

    public:
    __forceinline__ __device__ explicit SphInteractionManager(
        unsigned char *inSharedMemBuffer, const IndexType inMaxPartPerCell,
        const Point3D<IndexType> inNbCells)
        : bufferOfShared(inSharedMemBuffer), maxPartPerCell(inMaxPartPerCell),
          nbCells(inNbCells),
          nbCellsBlockCompute((nbCells.x + gridDim.x - 1) / gridDim.x)
    {
        sharedMemBuffer = bufferOfShared;
    }

    __forceinline__ __device__ auto getMaxNbParticles() const
    {
        return maxPartPerCell * (nbCellsBlockCompute + 2);
    }

    template <class ObjectType>
    __forceinline__ __device__ auto getSmBlock(const IndexType nbElements)
    {
        // Ensure alignement
        sharedMemBuffer += (sizeof(ObjectType) -
                            (size_t(sharedMemBuffer) % sizeof(ObjectType))) %
                           sizeof(ObjectType);
        unsigned char *oldPtr = sharedMemBuffer;
        sharedMemBuffer += nbElements * sizeof(ObjectType);
        return reinterpret_cast<ObjectType *>(oldPtr);
    }

    template <class FuncCopy, class FuncInit, class FuncCompute,
              class FuncRelease>
    __forceinline__ __device__ void computeInteraction(
        const RealType *inputX, const RealType *inputY, const RealType *inputZ,
        const IndexType *gridCellsCounts, const Point3D<RealType> inCellWidth,
        const Point3D<RealType> inBoxOrigin, FuncCopy &&copyLambda,
        FuncInit &&initLambda, FuncCompute &&computeLambda,
        FuncRelease &&releaseLambda)
    {
        assert(nbCells.y == gridDim.y);
        assert(nbCells.z == gridDim.z);

        const IndexType firstCellComputeGlobal =
            blockIdx.x * nbCellsBlockCompute;
        assert(firstCellComputeGlobal < nbCells.x);
        const IndexType lastCellComputeGlobal =
            min(nbCells.x, firstCellComputeGlobal + nbCellsBlockCompute);
        const IndexType nbLocalCellsCompute =
            lastCellComputeGlobal - firstCellComputeGlobal;
        assert(0 < nbLocalCellsCompute);

        // We load one more cell in each direction to compute the interaction
        const IndexType firstCellLoadGlobal =
            max(0, firstCellComputeGlobal - 1);
        const IndexType lastCellLoadGlobal =
            min(nbCells.x, lastCellComputeGlobal + 1);
        const IndexType nbLocalCellsLoad =
            lastCellLoadGlobal - firstCellLoadGlobal;
        // The number of threads must be greater than the number of cells we
        // load
        assert(nbLocalCellsCompute <= nbLocalCellsLoad &&
               nbLocalCellsLoad < blockDim.x);

        const IndexType firstCellLoadLocal = 0;
        const IndexType firstCellComputeLocal =
            firstCellComputeGlobal - firstCellLoadGlobal;
        const IndexType lastCellComputeLocal =
            lastCellComputeGlobal - firstCellLoadGlobal;

        // We load the offset array for the cells we will load
        IndexType *copygridCellsCounts =
            getSmBlock<IndexType>(nbLocalCellsLoad + 1);
        if (threadIdx.x <= nbLocalCellsLoad)
        {
            assert((blockIdx.z * nbCells.y + blockIdx.y) * nbCells.x +
                       firstCellLoadGlobal + threadIdx.x <=
                   nbCells.x * nbCells.y * nbCells.z);
            copygridCellsCounts[threadIdx.x] =
                gridCellsCounts[(blockIdx.z * nbCells.y + blockIdx.y) *
                                    nbCells.x +
                                firstCellLoadGlobal + threadIdx.x];
            assert(0 <= copygridCellsCounts[threadIdx.x]);
            assert(copygridCellsCounts[threadIdx.x] <=
                   gridCellsCounts[nbCells.x * nbCells.y * nbCells.z]);
            assert(threadIdx.x == 0 || copygridCellsCounts[threadIdx.x - 1] <=
                                           copygridCellsCounts[threadIdx.x]);
        }
        __syncthreads();

        const IndexType nbParticlesToCompute =
            copygridCellsCounts[lastCellComputeLocal] -
            copygridCellsCounts[firstCellComputeLocal];
        assert(0 <= nbParticlesToCompute);

        // If no particles to compute, then stop here
        if (nbParticlesToCompute == 0)
        {
            return;
        }

        const IndexType NbParticlesToLoad =
            copygridCellsCounts[nbLocalCellsLoad] -
            copygridCellsCounts[firstCellLoadLocal];

        assert(0 <= NbParticlesToLoad);
        assert(NbParticlesToLoad <= blockDim.x);
        assert(nbParticlesToCompute <= NbParticlesToLoad);
        assert(NbParticlesToLoad <= getMaxNbParticles());

        // Load values for all particles (in the load interval)
        Point3D<RealType> *particlePositions =
            getSmBlock<Point3D<RealType>>(getMaxNbParticles());

        if (threadIdx.x < NbParticlesToLoad)
        {
            const IndexType globalIdx =
                copygridCellsCounts[firstCellLoadLocal] + threadIdx.x;
            assert(globalIdx <
                   gridCellsCounts[nbCells.x * nbCells.y * nbCells.z]);

            particlePositions[threadIdx.x].x = inputX[globalIdx];
            particlePositions[threadIdx.x].y = inputY[globalIdx];
            particlePositions[threadIdx.x].z = inputZ[globalIdx];

            // Ask user to copy the needed data
            copyLambda(threadIdx.x, globalIdx);
        }
        __syncthreads();

        Point3D<IndexType> myCellGlobalIdx;
        IndexType myCellLocalIdxX = -1;
        Point3D<RealType> myParticlePos;
        IndexType myParticleGlobalIdx = -1;
        IndexType myParticleLocalIdx = -1;
        IndexType indexFirstCell = -1;
        IndexType indexLastCell = -1;
        if (threadIdx.x < nbParticlesToCompute)
        {
            // Copy the position from the shared memory
            myParticleLocalIdx = threadIdx.x +
                                 copygridCellsCounts[firstCellComputeLocal] -
                                 copygridCellsCounts[firstCellLoadLocal];
            myParticlePos = particlePositions[myParticleLocalIdx];
            // We do not know which cell the particle is in, we need to compute
            // it
            myCellGlobalIdx.x = static_cast<IndexType>(
                (myParticlePos.x - inBoxOrigin.x) / inCellWidth.x);
            myCellGlobalIdx.y = static_cast<IndexType>(
                (myParticlePos.y - inBoxOrigin.y) / inCellWidth.y);
            myCellGlobalIdx.z = static_cast<IndexType>(
                (myParticlePos.z - inBoxOrigin.z) / inCellWidth.z);
            assert(0 <= myCellGlobalIdx.x && myCellGlobalIdx.x < nbCells.x);
            assert(0 <= myCellGlobalIdx.y && myCellGlobalIdx.y < nbCells.y);
            assert(0 <= myCellGlobalIdx.z && myCellGlobalIdx.z < nbCells.z);
            assert(myCellGlobalIdx.y == blockIdx.y);
            assert(myCellGlobalIdx.z == blockIdx.z);

            // Compute the index in the shared memory
            myCellLocalIdxX = myCellGlobalIdx.x - firstCellLoadGlobal;
            assert(0 <= myCellLocalIdxX);
            assert(myCellLocalIdxX < nbLocalCellsLoad);
            myParticleGlobalIdx =
                threadIdx.x + copygridCellsCounts[firstCellComputeLocal];
            assert(myParticleGlobalIdx <
                   gridCellsCounts[nbCells.x * nbCells.y * nbCells.z]);
            assert(gridCellsCounts[(myCellGlobalIdx.z * nbCells.y +
                                    myCellGlobalIdx.y) *
                                       nbCells.x +
                                   myCellGlobalIdx.x] <= myParticleGlobalIdx);
            assert(myParticleGlobalIdx <
                   gridCellsCounts[(myCellGlobalIdx.z * nbCells.y +
                                    myCellGlobalIdx.y) *
                                       nbCells.x +
                                   myCellGlobalIdx.x + 1]);

            // Tell the user to copy the data for a given target particle
            initLambda(myParticleLocalIdx, myParticleGlobalIdx, myParticlePos);

            indexFirstCell =
                (myCellLocalIdxX == 0) ? myCellLocalIdxX : myCellLocalIdxX - 1;
            assert(0 <= indexFirstCell);
            const IndexType firstPart = copygridCellsCounts[indexFirstCell];
            indexLastCell = (myCellLocalIdxX + 1 < nbLocalCellsLoad)
                                ? myCellLocalIdxX + 2
                                : myCellLocalIdxX + 1;
            assert(indexLastCell <= nbLocalCellsLoad);
            assert(indexFirstCell < indexLastCell);
            const IndexType lastPart = copygridCellsCounts[indexLastCell];
            assert(0 <= firstPart);
            assert(firstPart < lastPart);
            assert(lastPart <=
                   gridCellsCounts[nbCells.x * nbCells.y * nbCells.z]);

            for (IndexType idxGlobalSrc = firstPart; idxGlobalSrc < lastPart;
                 ++idxGlobalSrc)
            {
                const IndexType idxLocalSrc =
                    idxGlobalSrc - copygridCellsCounts[firstCellLoadLocal];
                assert(0 <= idxLocalSrc && idxLocalSrc < NbParticlesToLoad);

                // Avoid self interaction
                if (myParticleLocalIdx != idxLocalSrc)
                {
                    computeLambda(myParticlePos, particlePositions[idxLocalSrc],
                                  idxLocalSrc, idxGlobalSrc);
                }
            }
        }
        __syncthreads();

        // Compute with current data
        for (IndexType idxZ = max(IndexType(blockIdx.z) - 1, 0);
             idxZ < min(IndexType(blockIdx.z) + 2, nbCells.z); ++idxZ)
        {
            for (IndexType idxY = max(IndexType(blockIdx.y) - 1, 0);
                 idxY < min(IndexType(blockIdx.y) + 2, nbCells.y); ++idxY)
            {
                if (idxZ != blockIdx.z || idxY != blockIdx.y)
                {
                    // Load values for all particles (in the load interval)
                    if (threadIdx.x <= nbLocalCellsLoad)
                    {
                        assert((idxZ * nbCells.y + idxY) * nbCells.x +
                                   firstCellLoadGlobal + threadIdx.x <=
                               nbCells.x * nbCells.y * nbCells.z);
                        copygridCellsCounts[threadIdx.x] =
                            gridCellsCounts[(idxZ * nbCells.y + idxY) *
                                                nbCells.x +
                                            firstCellLoadGlobal + threadIdx.x];

                        assert(0 <= copygridCellsCounts[threadIdx.x]);
                        assert(
                            copygridCellsCounts[threadIdx.x] <=
                            gridCellsCounts[nbCells.x * nbCells.y * nbCells.z]);
                        assert(threadIdx.x == 0 ||
                               copygridCellsCounts[threadIdx.x - 1] <=
                                   copygridCellsCounts[threadIdx.x]);
                    }
                    __syncthreads();

                    const IndexType NbParticlesToLoad =
                        copygridCellsCounts[nbLocalCellsLoad] -
                        copygridCellsCounts[firstCellLoadLocal];
                    assert(0 <= NbParticlesToLoad);
                    assert(NbParticlesToLoad <= getMaxNbParticles());

                    if (threadIdx.x < NbParticlesToLoad)
                    {
                        const IndexType globalIdx =
                            copygridCellsCounts[firstCellLoadLocal] +
                            threadIdx.x;
                        assert(
                            globalIdx <
                            gridCellsCounts[nbCells.x * nbCells.y * nbCells.z]);

                        particlePositions[threadIdx.x].x = inputX[globalIdx];
                        particlePositions[threadIdx.x].y = inputY[globalIdx];
                        particlePositions[threadIdx.x].z = inputZ[globalIdx];
                        copyLambda(threadIdx.x, globalIdx);
                    }

                    __syncthreads();

                    if (threadIdx.x < nbParticlesToCompute)
                    {
                        const IndexType firstPart =
                            copygridCellsCounts[indexFirstCell];
                        const IndexType lastPart =
                            copygridCellsCounts[indexLastCell];
                        assert(0 <= firstPart);
                        assert(firstPart <= lastPart);
                        assert(
                            lastPart <=
                            gridCellsCounts[nbCells.x * nbCells.y * nbCells.z]);

                        for (IndexType idxGlobalSrc = firstPart;
                             idxGlobalSrc < lastPart; ++idxGlobalSrc)
                        {
                            const IndexType idxLocalSrc =
                                idxGlobalSrc -
                                copygridCellsCounts[firstCellLoadLocal];
                            assert(0 <= idxLocalSrc &&
                                   idxLocalSrc < NbParticlesToLoad);

                            computeLambda(myParticlePos,
                                          particlePositions[idxLocalSrc],
                                          idxLocalSrc, idxGlobalSrc);
                        }
                    }
                    __syncthreads();
                }
            }
        }

        if (threadIdx.x < nbParticlesToCompute)
        {
            releaseLambda(myParticleGlobalIdx, myParticlePos);
        }
        __syncthreads();
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
            (nbrCells.x + expectedGridDim.x - 1) / expectedGridDim.x;

        const IndexType sizeOfIndexType = sizeof(IndexType);
        const IndexType sizeOfRealType = sizeof(RealType);

        const IndexType totalNbCellsInBlock = widthCellX + 2;
        const IndexType totalNbParticles = totalNbCellsInBlock * maxPartPerCell;

        // Corresponds to the memory used by the 2 offset array (local and
        // global) for all blocks and by the array to move from local to global
        // for each particles (the + 2 is here because offset array have one
        // additional element).
        const IndexType memorySizeForFirstTerm =
            sizeOfIndexType * (totalNbCellsInBlock + 1);

        // Corresponds to the memory used by the position (3*sizeOfRealType) of
        // each particles
        const IndexType memorySizeForSecondTerm =
            std::max(IndexType(0),
                     IndexType(sizeOfRealType - sizeOfIndexType)) +
            sizeOfRealType * totalNbParticles * 3 +
            totalNbParticles * additionalSizeSMPerParticle;

        const IndexType smNeededSize =
            memorySizeForFirstTerm + memorySizeForSecondTerm;

        static_assert(sizeof(Point3D<RealType>) == 3 * sizeof(RealType),
                      "Point3D must be a simple struct");

        assert(smNeededSize <= getSharedMemPerBlock());
        return smNeededSize;
    }

    /**
     * @brief      Determined the size of shared memory for one cell from the
     * max number of particles in one cell.
     *
     * @param[in]  maxPartPerCell  The theoretical maximum number of
     * particle per cell
     *
     * @return     The size of byte of shared memory for one cell.
     */
    __host__ static IndexType sharedMemSizeOneCell(
        const IndexType maxPartPerCell,
        const IndexType additionalSizeSMPerParticle = IndexType(0))
    {
        const IndexType totalNbCellsInBlock = 1;
        const IndexType totalNbParticles = totalNbCellsInBlock * maxPartPerCell;
        // Calculate the size of shared memory required for one cell
        const IndexType sizeOfIndexType = sizeof(IndexType);
        const IndexType sizeOfRealType = sizeof(RealType);

        // Corresponds to the memory used by the 2 offset array (local and
        // global) for all blocks and by the array to move from local to global
        // for each particles (the + 2 is here because offset array have one
        // additional element).
        const IndexType memorySizeForFirstTerm =
            sizeOfIndexType * (totalNbCellsInBlock + 1);

        // Corresponds to the memory used by the position (3*sizeOfRealType) of
        // each particles
        const IndexType memorySizeForSecondTerm =
            std::max(IndexType(0),
                     IndexType(sizeOfRealType - sizeOfIndexType)) +
            sizeOfRealType * totalNbParticles * 3 +
            totalNbParticles * additionalSizeSMPerParticle;

        const IndexType smNeededSize =
            memorySizeForFirstTerm + memorySizeForSecondTerm;

        assert(smNeededSize <= getSharedMemPerBlock());
        return smNeededSize;
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
        const IndexType additionalSizeSMPerParticle, const int divSM = 3)
    {
        const IndexType smSizePerCell = sharedMemSizeOneCell(
            maxParticlesPerCell, additionalSizeSMPerParticle);
        auto [enable, dimGrid, dimBlock] =
            getSharedmemConf(nbrCells, maxParticlesPerCell, smSizePerCell, divSM);
        if(enable == false){
            return std::make_tuple(false, int3{0, 0, 0}, 0, 0);
        }

        const IndexType smSize =
            sharedMemSize(nbrCells, dimGrid, maxParticlesPerCell,
                          additionalSizeSMPerParticle);

        return std::make_tuple(true, dimGrid, dimBlock, smSize);
    }
};

} // namespace manager_v2
