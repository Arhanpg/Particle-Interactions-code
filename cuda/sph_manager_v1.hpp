#pragma once
#include "point3d.hpp"

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <tuple>

namespace manager_v1
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

    // The maximum number of cells a block can manage due to the total number of
    // cells constraints.
    const int MaxNbCellsPerBlock = std::max(1, nbrTotalCells / getMultiProcessorCount());
    // The maximum number of cells a block can manage due to shared memory
    // constraints (including ghosts).
    const int MaxNbCellsPerMem = getSharedMemPerBlock() / memPerCell;
    if(MaxNbCellsPerMem < 3*3*3){ // We need at least 3x3x3 cells
        return std::make_tuple(false, int3{0, 0, 0}, 0);
    }
    // The width of the cube that respect the constraints
    // it includes the neighbors
    const int NbCellsPerBlock1Dim = std::min(int(cbrt(MaxNbCellsPerMem)), 
                                    int(cbrt(MaxNbCellsPerBlock))+2);

    assert(NbCellsPerBlock1Dim >= 3);

    int MaxCellPerBlockX = NbCellsPerBlock1Dim;
    int MaxCellPerBlockY = NbCellsPerBlock1Dim;
    int MaxCellPerBlockZ = NbCellsPerBlock1Dim;

    // We will see if we can increase the size in one dimension while respecting
    // the constraints
    if ((MaxCellPerBlockX + 1) * MaxCellPerBlockY * MaxCellPerBlockZ <= MaxNbCellsPerMem
        && ((MaxCellPerBlockX-2) + 1) * (MaxCellPerBlockY-2) * (MaxCellPerBlockZ-2) <= MaxNbCellsPerBlock)
    {
        MaxCellPerBlockX += 1;

        if (MaxCellPerBlockX * (MaxCellPerBlockY + 1) * MaxCellPerBlockZ <=  MaxNbCellsPerMem
            && (MaxCellPerBlockX-2) * ((MaxCellPerBlockY-2) + 1) * (MaxCellPerBlockZ-2) <= MaxNbCellsPerBlock)
        {
            MaxCellPerBlockY += 1;
        }
        else if ((MaxCellPerBlockX + 1) * MaxCellPerBlockY * MaxCellPerBlockZ <= MaxNbCellsPerMem
             && ((MaxCellPerBlockX-2) + 1) * (MaxCellPerBlockY-2) * (MaxCellPerBlockZ-2) <= MaxNbCellsPerBlock)
        {
            MaxCellPerBlockX += 1;
        }
    }
    assert(MaxCellPerBlockX >= 3 && MaxCellPerBlockY >= 3 &&
           MaxCellPerBlockZ >= 3);

    // We have included the neighbors cells and now we remove them
    MaxCellPerBlockX = MaxCellPerBlockX - 2;
    MaxCellPerBlockY = MaxCellPerBlockY - 2;
    MaxCellPerBlockZ = MaxCellPerBlockZ - 2;

    // We compute the number of blocks
    const int NbBlocksX =
        std::max(1, (simBoxGrid.x + MaxCellPerBlockX - 1) / MaxCellPerBlockX);
    const int NbBlocksY =
        std::max(1, (simBoxGrid.y + MaxCellPerBlockY - 1) / MaxCellPerBlockY);
    const int NbBlocksZ =
        std::max(1, (simBoxGrid.z + MaxCellPerBlockZ - 1) / MaxCellPerBlockZ);
    // Store it in a int3
    const int3 dimGrid{NbBlocksX, NbBlocksY, NbBlocksZ};

    // Compute the number of threads with min and max bound
    const int MaxNbThreads = 512;
    const int MinNbThreads = 32;

    // max number of particles per block (max number of particles per cell time
    // the number of cell per block in 3D)
    const int maxNbrParticlesPerBlock = MaxCellPerBlockX * MaxCellPerBlockY *
                                        MaxCellPerBlockZ * maxParticlesPerCell;
    // we make sure that the number of particles per block is greater than the
    // minimum number of threads
    const int nbrThreadPerBlock =
        std::max(MinNbThreads, maxNbrParticlesPerBlock);
    // Ideally the number of threads is the expected number of particles for the
    // number of cells per block, but we make sure it is between MinNbThreads
    // and MaxNbThreads.
    const int NbThreads = std::min(MaxNbThreads, nbrThreadPerBlock);
    return std::make_tuple(true, dimGrid, NbThreads);
}

void kernelCallerCudaMemcpy3FToF3(float3 *dest, const float *srcX,
                                  const float *srcY, const float *srcZ,
                                  size_t count);

template <typename IndexType>
__forceinline__ __device__ __host__ int log2OfPowerOfTwo(
    const IndexType inPowerOfTwo)
{
    assert(inPowerOfTwo > 0);
#ifdef __CUDA_ARCH__
    return __ffs(static_cast<size_t>(inPowerOfTwo))-1;
#else
    return ffs(static_cast<size_t>(inPowerOfTwo))-1;
#endif
}

template <typename ValueType>
__forceinline__ __device__ __host__ ValueType
greatestOrEqualPowerOf2(ValueType inValue)
{
    static_assert(std::is_integral<ValueType>::value,
                  "Must be an integral type");

    if (inValue == 0)
    {
        return 1;
    }

    if ((inValue & (inValue - 1)) == 0)
    {
        // inValue is already a power of 2
        return inValue;
    }

    int nbBits = sizeof(int64_t) * 8;
#ifdef __CUDA_ARCH__
    const int nbLeading0 = __clzll(static_cast<int64_t>(inValue));
#else
    const int nbLeading0 = __builtin_clzl(static_cast<int64_t>(inValue));
#endif
    return static_cast<ValueType>(1) << (nbBits - nbLeading0);
}

template <typename NumType, typename IndexType>
__forceinline__ __device__ void buildPrefixSMDevice(NumType *prefixTree,
                                                    const IndexType inSize)
{
    const auto powerOfTwo = greatestOrEqualPowerOf2(size_t(inSize));
    const IndexType TreeHeight = log2OfPowerOfTwo(powerOfTwo);

    {
        // Upward pass
        IndexType jumpSize = 2;

        for (IndexType idxLevel = TreeHeight - 1; idxLevel >= 0; --idxLevel)
        {
            const IndexType jumpSizeDiv2 = jumpSize / 2;
            for (IndexType idxNode = threadIdx.x * jumpSize + jumpSize - 1;
                 idxNode < inSize; idxNode += blockDim.x * jumpSize)
            {
                prefixTree[idxNode] += prefixTree[idxNode - jumpSizeDiv2];
            }
            jumpSize *= 2;
            __syncthreads();
        }
    }

    {
        // Downward pass
        IndexType jumpSize =
            (greatestOrEqualPowerOf2(size_t(inSize)) >> (2 - 1));

        for (IndexType idxLevel = 1; idxLevel < TreeHeight; ++idxLevel)
        {
            const IndexType jumpSizeDiv2 = jumpSize / 2;
            for (IndexType idxNode =
                     threadIdx.x * jumpSize + jumpSize + jumpSizeDiv2 - 1;
                 idxNode < inSize; idxNode += blockDim.x * jumpSize)
            {
                prefixTree[idxNode] += prefixTree[idxNode - jumpSizeDiv2];
            }
            jumpSize /= 2;
            __syncthreads();
        }
    }
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
    unsigned char *const bufferOfShared;
    unsigned char *ptrToBufferOfSharedMem;

    IndexType firstCellX;
    IndexType lastCellX;
    IndexType limiteSubX;
    IndexType nbLocalCellsX;
    IndexType firstCellY;
    IndexType lastCellY;
    IndexType limiteSubY;
    IndexType nbLocalCellsY;
    IndexType firstCellZ;
    IndexType lastCellZ;
    IndexType limiteSubZ;
    IndexType nbLocalCellsZ;
    IndexType totalNbCells;

    IndexType totalNbParticles;
    IndexType *__restrict__ localCellOffset;
    IndexType *__restrict__ globalCellOffset;
    IndexType *__restrict__ locToGlob;

    // Needed for the computation:
    RealType *__restrict__ srcX;
    RealType *__restrict__ srcY;
    RealType *__restrict__ srcZ;

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
    __forceinline__ __device__ bool init(
        const RealType *const __restrict__ positionsParticlesX,
        const RealType *const __restrict__ positionsParticlesY,
        const RealType *const __restrict__ positionsParticlesZ,
        const IndexType *const __restrict__ gridCellsCounts,
        const Point3D<IndexType> nbCells)
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
        this->firstCellX = blockIdx.x * nbCellsBlockX;
        this->lastCellX = min(nbCells.x, this->firstCellX + nbCellsBlockX);
        this->limiteSubX = max(0, this->firstCellX - extraStep);
        const IndexType limiteUpX = min(nbCells.x, this->lastCellX + extraStep);
        this->nbLocalCellsX = limiteUpX - this->limiteSubX;

        this->firstCellY = blockIdx.y * nbCellsBlockY;
        this->lastCellY = min(nbCells.y, this->firstCellY + nbCellsBlockY);
        this->limiteSubY = max(0, this->firstCellY - extraStep);
        const IndexType limiteUpY = min(nbCells.y, this->lastCellY + extraStep);
        this->nbLocalCellsY = limiteUpY - this->limiteSubY;

        this->firstCellZ = blockIdx.z * nbCellsBlockZ;
        this->lastCellZ = min(nbCells.z, this->firstCellZ + nbCellsBlockZ);
        this->limiteSubZ = max(0, this->firstCellZ - extraStep);
        const IndexType limiteUpZ = min(nbCells.z, this->lastCellZ + extraStep);
        this->nbLocalCellsZ = limiteUpZ - this->limiteSubZ;
        assert(this->nbLocalCellsX > 0 && this->nbLocalCellsY > 0 &&
               this->nbLocalCellsZ > 0);

        // The total number of cells we will need to load data from
        this->totalNbCells = nbLocalCellsX * nbLocalCellsY * nbLocalCellsZ;

        // Check if the box is empty
        __shared__ bool emptyCells;
        if (threadIdx.x == 0)
        {
            const IndexType expectedNbParticles =
                (gridCellsCounts[((this->lastCellZ) * nbCells.y +
                                  this->lastCellY) *
                                     nbCells.x +
                                 this->lastCellX]) -
                (gridCellsCounts[((this->firstCellZ) * nbCells.y +
                                  this->firstCellY) *
                                     nbCells.x +
                                 this->firstCellX]);
            emptyCells = (expectedNbParticles == 0);
        }

        __syncthreads();

        if (emptyCells)
        {
            return false;
        }
        this->localCellOffset = getSmBlock<IndexType>(this->totalNbCells + 1);
        this->globalCellOffset = getSmBlock<IndexType>(this->totalNbCells + 1);

        {
            // We start by copying the offset for each of the cells
            for (IndexType idxCell = threadIdx.x; idxCell < this->totalNbCells;
                 idxCell += blockDim.x)
            {
                // Retrieve the coordinate of the cell
                const IndexType idxX = idxCell % this->nbLocalCellsX;
                const IndexType idxY =
                    (idxCell / this->nbLocalCellsX) % this->nbLocalCellsY;
                const IndexType idxZ =
                    idxCell / (this->nbLocalCellsX * this->nbLocalCellsY);
                assert((idxZ * this->nbLocalCellsY + idxY) *
                               this->nbLocalCellsX +
                           idxX ==
                       idxCell);
                globalCellOffset[idxCell] =
                    gridCellsCounts[((idxZ + this->limiteSubZ) * nbCells.y +
                                     idxY + this->limiteSubY) *
                                        nbCells.x +
                                    idxX + this->limiteSubX];
            }

            if (this->totalNbCells % blockDim.x == threadIdx.x)
            {
                const int idxCell = this->totalNbCells - 1;
                const IndexType idxX = idxCell % this->nbLocalCellsX;
                const IndexType idxY =
                    (idxCell / this->nbLocalCellsX) % this->nbLocalCellsY;
                const IndexType idxZ =
                    idxCell / (this->nbLocalCellsX * this->nbLocalCellsY);
                globalCellOffset[idxCell + 1] =
                    gridCellsCounts[((idxZ + this->limiteSubZ) * nbCells.y +
                                     idxY + this->limiteSubY) *
                                        nbCells.x +
                                    idxX + this->limiteSubX + 1];
            }
            __syncthreads();

            // We now compute the number of particles to skip from one x-pencil
            // to the other
            const IndexType nbCellBorder =
                this->nbLocalCellsY * this->nbLocalCellsZ;
            /*__shared__*/ IndexType *__restrict__ offsetGlobalToLobal =
                reinterpret_cast<IndexType *>(this->ptrToBufferOfSharedMem);
            // Not needed, we will leave this mem available after this scope
            // getSmBlock<IndexType>(this->nbCellBorder);
            // Special case for the first element
            if (threadIdx.x == 0)
            {
                offsetGlobalToLobal[0] = globalCellOffset[0];
            }
            for (IndexType idxBorderCell = threadIdx.x + 1;
                 idxBorderCell < nbCellBorder; idxBorderCell += blockDim.x)
            {
                const IndexType idxY = idxBorderCell % this->nbLocalCellsY;
                const IndexType idxZ = idxBorderCell / this->nbLocalCellsY;
                assert(idxY || idxZ);
                // We compute the position of the previous x-pencil
                const IndexType idxYO =
                    (idxY == 0) ? this->nbLocalCellsY - 1 : idxY - 1;
                const IndexType idxZO = (idxY == 0) ? idxZ - 1 : idxZ;
                // We need to substract from the first case of the x-pencil
                // the value of the case after the previous x-pencil
                // it could be that nbLocalCellsX+this->limiteSubX == nbCells.x
                // it would be fine, we would go on the value we want anyway
                offsetGlobalToLobal[idxBorderCell] =
                    globalCellOffset[(idxZ * this->nbLocalCellsY + idxY) *
                                     this->nbLocalCellsX] -
                    gridCellsCounts[((idxZO + this->limiteSubZ) * nbCells.y +
                                     idxYO + this->limiteSubY) *
                                        nbCells.x +
                                    this->nbLocalCellsX + this->limiteSubX];
            }
            __syncthreads();

            // We have a 2D matrix of number of particle to remove from one
            // x-stencil to the other we need to prefix sum them
            buildPrefixSMDevice(offsetGlobalToLobal, nbCellBorder);
            __syncthreads();

            // The local cell offset is the global offset minus the particles to
            // exclude
            for (IndexType idxCell = threadIdx.x; idxCell < this->totalNbCells;
                 idxCell += blockDim.x)
            {
                const IndexType idxX = idxCell % this->nbLocalCellsX;
                const IndexType idxY =
                    (idxCell / this->nbLocalCellsX) % this->nbLocalCellsY;
                const IndexType idxZ =
                    idxCell / (this->nbLocalCellsX * this->nbLocalCellsY);

                localCellOffset[(idxZ * this->nbLocalCellsY + idxY) *
                                    this->nbLocalCellsX +
                                idxX] =
                    globalCellOffset[(idxZ * this->nbLocalCellsY + idxY) *
                                         this->nbLocalCellsX +
                                     idxX] -
                    offsetGlobalToLobal[idxZ * this->nbLocalCellsY + idxY];
            }

            // Special case for the last element of the prefix sum
            if (this->totalNbCells % blockDim.x == threadIdx.x)
            {
                const IndexType idxCell = this->totalNbCells - 1;
                const IndexType idxY =
                    (idxCell / this->nbLocalCellsX) % this->nbLocalCellsY;
                const IndexType idxZ =
                    idxCell / (this->nbLocalCellsX * this->nbLocalCellsY);
                localCellOffset[this->totalNbCells] =
                    globalCellOffset[this->totalNbCells] -
                    offsetGlobalToLobal[idxZ * this->nbLocalCellsY + idxY];
            }
            __syncthreads();
        }


        // The total number of particles is the last value of the local offset
        // table
        this->totalNbParticles = localCellOffset[this->totalNbCells];

        // assert(this->totalNbParticles <=
        //        this->totalNbCells * TMP_MAX_PART_PER_CELL);
        this->locToGlob = getSmBlock<IndexType>(this->totalNbParticles);

        // We first need to compute the position of each particle in the local
        // array
        for (IndexType idxCell = threadIdx.x; idxCell < this->totalNbCells;
             idxCell += blockDim.x)
        {
            const IndexType idxX = idxCell % this->nbLocalCellsX;
            const IndexType idxY =
                (idxCell / this->nbLocalCellsX) % this->nbLocalCellsY;
            const IndexType idxZ =
                idxCell / (this->nbLocalCellsX * this->nbLocalCellsY);
            assert(idxCell ==
                   (idxZ * this->nbLocalCellsY + idxY) * this->nbLocalCellsX +
                       idxX);

            // For each cell we find out the first and last particle positions
            // in the table
            const IndexType idxFirstPart = localCellOffset[idxCell];
            const IndexType idxLastPart = localCellOffset[idxCell + 1];

            // From here we are able to compute the local to global mapping
            assert(idxFirstPart <= idxLastPart);
            for (IndexType idxPartInCell = idxFirstPart;
                 idxPartInCell < idxLastPart; ++idxPartInCell)
            {
                this->locToGlob[idxPartInCell] = idxPartInCell -
                                                 localCellOffset[idxCell] +
                                                 globalCellOffset[idxCell];
            }
        }
        __syncthreads();

        return true;
    }

    __forceinline__ __device__ IndexType getNbParticles() const
    {
        return this->totalNbParticles;
    }

    /**
     * @brief      Copy the position to shared memory and apply function inFunc
     * that is dedicated to copy additional content.
     *
     * @param[in]  positionsParticlesX  The positions particles x
     * @param[in]  positionsParticlesY  The positions particles y
     * @param[in]  positionsParticlesZ  The positions particles z
     * @param      inFunc               Lambda function to copy additional
     * content to shared memory
     */
    template <class Func>
    __forceinline__ __device__ void bufferBuildAndCopy(
        const RealType *const __restrict__ positionsParticlesX,
        const RealType *const __restrict__ positionsParticlesY,
        const RealType *const __restrict__ positionsParticlesZ, Func &&inFunc)
    {
        this->srcX = getSmBlock<RealType>(this->totalNbParticles);
        this->srcY = getSmBlock<RealType>(this->totalNbParticles);
        this->srcZ = getSmBlock<RealType>(this->totalNbParticles);

        // We copy the position and subtract the corner to factorize this work
        for (IndexType idxPart = threadIdx.x; idxPart < this->totalNbParticles;
             idxPart += blockDim.x)
        {
            const IndexType globalOffset = this->locToGlob[idxPart];
            this->srcX[idxPart] = positionsParticlesX[globalOffset];
            this->srcY[idxPart] = positionsParticlesY[globalOffset];
            this->srcZ[idxPart] = positionsParticlesZ[globalOffset];

            // Kernel can copy whatever he wants
            inFunc(idxPart, globalOffset);
        }

        __syncthreads();
    }

    /**
     * @brief      Main core of particles calculation. For a particles i, it
     * will, using shared memory, compute the interaction between i and its
     * neighbors.
     *
     * @param[in]  inCellWidth       The cell width
     * @param[in]  inBoxOrigin       The box origin
     * @param      FuncInit          The function initialize
     * @param      FuncInteract      The function interact
     * @param      FuncRelease       The function release
     *
     */
    template <class FuncInitType, class FuncInteractType, class FuncReleaseType>
    __forceinline__ __device__ void computeInteraction(
        const Point3D<RealType> inCellWidth,
        const Point3D<RealType> inBoxOrigin, FuncInitType &&FuncInit,
        FuncInteractType &&FuncInteract, FuncReleaseType &&FuncRelease)
    {
        for (IndexType idxPart = threadIdx.x; idxPart < this->totalNbParticles;
             idxPart += blockDim.x)
        {
            const IndexType localIdx = idxPart;
            const IndexType globalIdx = locToGlob[idxPart];

            const Point3D<RealType> xI{this->srcX[localIdx],
                                       this->srcY[localIdx],
                                       this->srcZ[localIdx]};

            const bool willInteract = FuncInit(idxPart, globalIdx, xI);
            if (!willInteract)
            {
                continue;
            }

            // Get global cell coordinate
            IndexType idxSubCellX =
                static_cast<IndexType>((xI.x - inBoxOrigin.x) / inCellWidth.x);
            IndexType idxSubCellY =
                static_cast<IndexType>((xI.y - inBoxOrigin.y) / inCellWidth.y);
            IndexType idxSubCellZ =
                static_cast<IndexType>((xI.z - inBoxOrigin.z) / inCellWidth.z);

            // Must be in the valid interval
            assert(this->limiteSubX <= idxSubCellX &&
                   idxSubCellX < this->limiteSubX + this->nbLocalCellsX);
            assert(this->limiteSubY <= idxSubCellY &&
                   idxSubCellY < this->limiteSubY + this->nbLocalCellsY);
            assert(this->limiteSubZ <= idxSubCellZ &&
                   idxSubCellZ < this->limiteSubZ + this->nbLocalCellsZ);

            if (idxSubCellX < this->firstCellX ||
                this->lastCellX <= idxSubCellX ||
                idxSubCellY < this->firstCellY ||
                this->lastCellY <= idxSubCellY ||
                idxSubCellZ < this->firstCellZ ||
                this->lastCellZ <= idxSubCellZ)
            {
                // We are on a ghost cell, just skip it
                continue;
            }

            // Get local cell coordinate
            idxSubCellX -= this->limiteSubX;
            idxSubCellY -= this->limiteSubY;
            idxSubCellZ -= this->limiteSubZ;

            // Get local neighbor intervals
            const IndexType idxNeighXLimitSub = max(0, idxSubCellX - 1);
            const IndexType idxNeighXLimitUp =
                min(this->nbLocalCellsX, idxSubCellX + 2);
            const IndexType idxNeighYLimitSub = max(0, idxSubCellY - 1);
            const IndexType idxNeighYLimitUp =
                min(this->nbLocalCellsY, idxSubCellY + 2);
            const IndexType idxNeighZLimitSub = max(0, idxSubCellZ - 1);
            const IndexType idxNeighZLimitUp =
                min(this->nbLocalCellsZ, idxSubCellZ + 2);

            // loop on all neighbors cells
            for (IndexType idxNeighZ = idxNeighZLimitSub;
                 idxNeighZ < idxNeighZLimitUp; ++idxNeighZ)
            {
                for (IndexType idxNeighY = idxNeighYLimitSub;
                     idxNeighY < idxNeighYLimitUp; ++idxNeighY)
                {
                    for (IndexType idxNeighX = idxNeighXLimitSub;
                         idxNeighX < idxNeighXLimitUp; ++idxNeighX)
                    {
                        const IndexType idxSrcCell =
                            (idxNeighZ * this->nbLocalCellsY + idxNeighY) *
                                this->nbLocalCellsX +
                            idxNeighX;
                        for (IndexType idxPartSrc =
                                 this->localCellOffset[idxSrcCell];
                             idxPartSrc < this->localCellOffset[idxSrcCell + 1];
                             ++idxPartSrc)
                        {
                            // Test to make sure it is not the same part
                            if (idxPart != idxPartSrc)
                            {
                                const Point3D<RealType> xJ{
                                    this->srcX[idxPartSrc],
                                    this->srcY[idxPartSrc],
                                    this->srcZ[idxPartSrc]};
                                FuncInteract(xI, xJ, idxPartSrc,
                                             this->locToGlob[idxPartSrc]);
                            }
                        }
                    }
                }
            }

            FuncRelease(idxPart, globalIdx, xI);
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
        const IndexType totalNbParticles = totalNbCellsInBlock * maxPartPerCell;
        const IndexType nbCellsLeftSideBlock = widthCellY * widthCellZ;

        // Corresponds to the memory used by the 2 offset array (local and
        // global) for all blocks and by the array to move from local to global
        // for each particles (the + 2 is here because offset array have one
        // additional element).
        const IndexType memorySizeForFirstTerm =
            sizeOfIndexType * (totalNbCellsInBlock * 2 + 2 + totalNbParticles);

        // Corresponds to the memory used by the position (3*sizeOfRealType) of
        // each particles
        const IndexType memorySizeForSecondTerm =
            std::max(IndexType(0),
                     IndexType(sizeOfRealType - sizeOfIndexType)) +
            sizeOfRealType * totalNbParticles *
                (3 + additionalSizeSMPerParticle);

        const IndexType memorySizeForThirdTerm =
            sizeOfIndexType *
            (totalNbCellsInBlock * 2 + 2 + nbCellsLeftSideBlock);

        const IndexType smNeededSize =
            memorySizeForFirstTerm +
            std::max(memorySizeForSecondTerm, memorySizeForThirdTerm);

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

        // Corresponds to the memory used by the 2 offset array (local and
        // global) for this block and by the array to moving from local to
        // global for each particles of the cells (the + 2 is here because
        // offset array have one additional element). eg. for
        // maxParticlesPerCell = 60, we get : 256
        const IndexType memorySizeForFirstTerm =
            sizeOfIndexType * (totalCellsInBlock * 2 + 2 +
                               totalCellsInBlock * maxParticlesPerCell);

        // Corresponds to the memory used by the position (3*sizeOfRealType) of
        // each particles in one cell with additional bytes for each particles
        // which is specific to kernel
        // eg. for maxParticlesPerCell = 60, we get : 720
        const IndexType memorySizeForSecondTerm =
            std::max(IndexType(0),
                     IndexType(sizeOfRealType - sizeOfIndexType)) +
            sizeOfRealType * maxParticlesPerCell *
                (totalCellsInBlock * 3 + additionalSizeSMPerParticle);

        // eg. for maxParticlesPerCell = 60, we get : 20
        const IndexType memorySizeForThirdTerm =
            sizeOfIndexType *
            (totalCellsInBlock * 2 + 2 + numCellsLeftSideBlock);

        // eg. for maxParticlesPerCell = 60, we get : 976
        return memorySizeForFirstTerm +
               std::max(memorySizeForSecondTerm, memorySizeForThirdTerm);
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