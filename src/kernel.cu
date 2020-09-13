#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// For clarity, to help keep track of which numbers mean what.
#define BoidIndex int       // Boid ID
#define CellIndex int       // Cell ID corresponding to Boid
#define ArrayIndex int      // indicates that this counter
                            // is NOT associated with a Boid or Cell,
                            // but generally a place in an array.

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 4.5f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

#define maxNeighborDist imax(imax(rule1Distance, rule2Distance), rule3Distance)

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);
int numBlocks;

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
// An index value accessing any of these two arrays is NOT a BoidIndex or CellIndex.
// It is purely for traversing through chunks of cells

BoidIndex *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
                                     // Married to dev_particleGridIndices, represents boid IDs / BoidIndex
CellIndex *dev_particleGridIndices;  // What grid cell is this particle in?
                                     // Married to dev_particleArrayIndices, represents cell IDs / CellIndex

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

glm::vec3* dev_pos2;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  numBlocks = ceil(numObjects / blockSize);
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");
    
  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(BoidIndex));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(CellIndex));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  cudaMalloc((void**)&dev_pos2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_posCopy failed!");

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/

__device__ glm::vec3 rule1(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    glm::vec3 perceivedCenter(0.0f, 0.0f, 0.0f);
    int numNeighbors = 0;

    for (int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule1Distance) {
            perceivedCenter += pos[i];
            numNeighbors++;
        }
    }

    if (numNeighbors > 0) {
        perceivedCenter /= numNeighbors;
    }

    return (perceivedCenter - pos[iSelf]) * rule1Scale;
}

__device__ glm::vec3 rule2(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    glm::vec3 c(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule2Distance) {
            c -= (pos[i] - pos[iSelf]);
        }
    }

    return c * rule2Scale;
}

__device__ glm::vec3 rule3(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {
    glm::vec3 perceivedVelocity(0.0f, 0.0f, 0.0f);
    int numNeighbors = 0;

    for (int i = 0; i < N; i++) {
        if (i != iSelf && glm::distance(pos[iSelf], pos[i]) < rule3Distance) {
            perceivedVelocity += vel[i];
            numNeighbors++;
        }
    }

    if (numNeighbors > 0) {
        perceivedVelocity /= numNeighbors;
    }

    return perceivedVelocity * rule3Scale;

}

__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

    glm::vec3 rule1Vel = rule1(N, iSelf, pos, vel),
              rule2Vel = rule2(N, iSelf, pos, vel),
              rule3Vel = rule3(N, iSelf, pos, vel);

    return rule1Vel + rule2Vel + rule3Vel;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its velocity based on its current position and velocity.
*/
__device__ glm::vec3 clampSpeed(glm::vec3 inputSpeed) {
    float currentSpeed = inputSpeed.length();
    float clampedSpeed = glm::clamp(currentSpeed, 0.0f, maxSpeed);
    return glm::normalize(inputSpeed) * clampedSpeed;
}

__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {

    // Compute a new velocity based on pos and vel1
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    glm::vec3 velChange = computeVelocityChange(N, index, pos, vel1);

    glm::vec3 resultVel = vel1[index] + velChange;

    // Clamp the speed
    resultVel = clampSpeed(resultVel);

    // Record the new velocity into vel2. Question: why NOT vel1?
    vel2[index] = resultVel;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__device__ glm::vec3 clampXYZForBounds(glm::vec3 xyz,
    int gridResolution) {
    
    glm::vec3 res(xyz);

    if (res.x < 0) {
        res.x = 0;
    }
    else if (res.x >= gridResolution) {
        res.x = gridResolution - 1;
    }

    if (res.y < 0) {
        res.y = 0;
    }
    else if (res.y >= gridResolution) {
        res.y = gridResolution - 1;
    }

    if (res.z < 0) {
        res.z = 0;
    }
    else if (res.z >= gridResolution) {
        res.z = gridResolution - 1;
    }

    return res;
}

__device__ glm::vec3 getCellXYZAtPos(glm::vec3 pos,
    int gridResolution, glm::vec3 gridMin, float inverseCellWidth) {

    int cellx = floor((pos.x - gridMin.x) * inverseCellWidth),
        celly = floor((pos.y - gridMin.y) * inverseCellWidth),
        cellz = floor((pos.z - gridMin.z) * inverseCellWidth);

    return glm::vec3(cellx, celly, cellz);

}

__device__ int getCellIndexAtPos(glm::vec3 pos, int gridResolution,
    glm::vec3 gridMin, float inverseCellWidth) {
   
    glm::vec3 cellXYZ = getCellXYZAtPos(pos, gridResolution,
                                         gridMin, inverseCellWidth);

    return gridIndex3Dto1D((int)cellXYZ.x, (int)cellXYZ.y, (int)cellXYZ.z, gridResolution);

}

__device__ glm::vec3 getXYZFromCellNumber(CellIndex index, int gridResolution, int gridCellCount) {
    if (index < 0 || index >= gridCellCount) {
        return glm::vec3(-1);
    }

    int z = index / (gridResolution * gridResolution);
    index -= (z * gridResolution * gridResolution);
    int y = index / gridResolution;
    int x = index % gridResolution;
    return glm::vec3(x, y, z);
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

    // Find the index of this thread, which corresponds to the calculations of the boid
    // with that index.
    BoidIndex index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index < N) {
        // Setup data for that boid.
        glm::vec3 boidPos = pos[index];
        gridIndices[index] = getCellIndexAtPos(boidPos, gridResolution, gridMin, inverseCellWidth);
        indices[index] = index;
    }

}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  // Get unique thread ID.
  ArrayIndex index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index >= N) {
      return;
  }

  // Get cell indices at thread ID and ID + 1. Compare.
  CellIndex thisCellIndex = particleGridIndices[index];

  if (index == 0) {
      gridCellStartIndices[thisCellIndex] = index;
  }
  
  if (index == N - 1) {
      gridCellEndIndices[thisCellIndex] = index;
      return;
  }

  CellIndex nextCellIndex = particleGridIndices[index + 1];

  if (nextCellIndex != thisCellIndex) {
    gridCellEndIndices[thisCellIndex] = index;
    gridCellStartIndices[nextCellIndex] = index + 1;
  }

}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

    // Identify grid cell where particle is in
    BoidIndex thisBoidIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thisBoidIndex >= N) {
        return;
    }

    glm::vec3 boidPos = pos[thisBoidIndex];
    glm::vec3 thisCellXYZ = getCellXYZAtPos(boidPos, gridResolution, gridMin, inverseCellWidth);

    // Find neighbors by getting the constraints of the cube, then find index for
    // all sets of coordinates,
    glm::vec3 minPoint(boidPos - glm::vec3(maxNeighborDist)),
              maxPoint(boidPos + glm::vec3(maxNeighborDist));
    glm::vec3 minCorner = getCellXYZAtPos(minPoint, gridResolution, gridMin, inverseCellWidth),
              maxCorner = getCellXYZAtPos(maxPoint, gridResolution, gridMin, inverseCellWidth);

    minCorner = clampXYZForBounds(minCorner, gridResolution);
    maxCorner = clampXYZForBounds(maxCorner, gridResolution);

    // Putting them in separate ints to save from (int) casts
    // during every loop

    int xMin = minCorner.x,
        xMax = maxCorner.x,
        yMin = minCorner.y,
        yMax = maxCorner.y,
        zMin = minCorner.z,
        zMax = maxCorner.z;

    glm::vec3 perceivedCenterR1(0.0f, 0.0f, 0.0f);
    int numNeighborsR1 = 0;

    glm::vec3 cR2(0.0f, 0.0f, 0.0f);

    glm::vec3 perceivedVelocityR3(0.0f, 0.0f, 0.0f);
    int numNeighborsR3 = 0;
    
    for (int z = zMin; z <= zMax; z++) {
        for (int y = yMin; y <= yMax; y++) {
            for (int x = xMin; x <= xMax; x++) {
                CellIndex currentCellIndex = gridIndex3Dto1D(x, y, z, gridResolution);
                
                for (ArrayIndex i = gridCellStartIndices[currentCellIndex];
                                i <= gridCellEndIndices[currentCellIndex];
                                i++) {

                    if (i < 0) {
                        continue;
                    }

                    BoidIndex boid = particleArrayIndices[i];
                    if (boid == thisBoidIndex) {
                        continue;
                    }

                    // Calculate the values using the boid neighbors for
                    // all three rules at the time.

                    // Rule 1
                    if (glm::distance(pos[thisBoidIndex], pos[boid]) < rule1Distance) {
                        perceivedCenterR1 += pos[boid];
                        numNeighborsR1++;
                    }

                    // Rule 2
                    if (glm::distance(pos[thisBoidIndex], pos[boid]) < rule2Distance) {
                        cR2 -= (pos[boid] - pos[thisBoidIndex]);
                    }

                    // Rule 3
                    if (glm::distance(pos[thisBoidIndex], pos[boid]) < rule3Distance) {
                        perceivedVelocityR3 += vel1[boid];
                        numNeighborsR3++;
                    }

                }
            }
        }
    }


    if (numNeighborsR1 > 0) {
        perceivedCenterR1 /= numNeighborsR1;
    }

    if (numNeighborsR3 > 0) {
        perceivedVelocityR3 /= numNeighborsR3;
    }

    glm::vec3 velChange = ((perceivedCenterR1 - pos[thisBoidIndex]) * rule1Scale)
                          + (cR2 * rule2Scale)
                          + (perceivedVelocityR3 * rule3Scale);

    glm::vec3 resultVel = vel1[thisBoidIndex] + velChange;

    // Clamp the speed
    resultVel = clampSpeed(resultVel);

    // Record the new velocity into vel2.
    vel2[thisBoidIndex] = resultVel;

}

#define squared(v) v * v

__device__ bool cellInRadius(glm::vec3 xyz, glm::vec3 origin, float radius,
                             float gridCellWidth) {
    glm::vec3 bottomCnr(xyz * gridCellWidth);
    glm::vec3 topCnr((xyz + glm::vec3(1) * gridCellWidth));

    float radiusSq = squared(radius);

    if (origin.x - bottomCnr.x) {
        radiusSq -= squared(origin.x - bottomCnr.x);
    } else if (origin.x > topCnr.x) {
        radiusSq -= squared(origin.x - topCnr.x);
    }
    
    if (origin.y - bottomCnr.y) {
        radiusSq -= squared(origin.y - bottomCnr.y);
    }
    else if (origin.y > topCnr.y) {
        radiusSq -= squared(origin.y - topCnr.y);
    }
    
    if (origin.z - bottomCnr.z) {
        radiusSq -= squared(origin.z - bottomCnr.z);
    }
    else if (origin.x > topCnr.x) {
        radiusSq -= squared(origin.z - topCnr.z);
    }

    return radiusSq > 0;
    
}

__global__ void kernShufflePosVelBuffers(int N, int *particleArrayIndices,
    glm::vec3 *pos1, glm::vec3* pos2,
    glm::vec3 *vel1, glm::vec3 *vel2) {
    ArrayIndex index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= N) {
        return;
    }

    BoidIndex thisBoidIndex = particleArrayIndices[index];
    pos2[index] = pos1[thisBoidIndex];
    vel2[index] = vel1[thisBoidIndex];

}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

     // Identify grid cell where particle is in
    BoidIndex thisBoidIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (thisBoidIndex >= N) {
        return;
    }

    glm::vec3 boidPos = pos[thisBoidIndex];
    glm::vec3 thisCellXYZ = getCellXYZAtPos(boidPos, gridResolution, gridMin, inverseCellWidth);

    // Find neighbors by getting the constraints of the cube, then find index for
    // all sets of coordinates,
    glm::vec3 minPoint(boidPos - glm::vec3(maxNeighborDist)),
        maxPoint(boidPos + glm::vec3(maxNeighborDist));
    glm::vec3 minCorner = getCellXYZAtPos(minPoint, gridResolution, gridMin, inverseCellWidth),
        maxCorner = getCellXYZAtPos(maxPoint, gridResolution, gridMin, inverseCellWidth);

    minCorner = clampXYZForBounds(minCorner, gridResolution);
    maxCorner = clampXYZForBounds(maxCorner, gridResolution);

    // Putting them in separate ints to save from (int) casts
    // during every loop

    int xMin = minCorner.x,
        xMax = maxCorner.x,
        yMin = minCorner.y,
        yMax = maxCorner.y,
        zMin = minCorner.z,
        zMax = maxCorner.z;

    glm::vec3 perceivedCenterR1(0.0f, 0.0f, 0.0f);
    int numNeighborsR1 = 0;

    glm::vec3 cR2(0.0f, 0.0f, 0.0f);

    glm::vec3 perceivedVelocityR3(0.0f, 0.0f, 0.0f);
    int numNeighborsR3 = 0;

    for (int z = zMin; z <= zMax; z++) {
        for (int y = yMin; y <= yMax; y++) {
            for (int x = xMin; x <= xMax; x++) {
                CellIndex currentCellIndex = gridIndex3Dto1D(x, y, z, gridResolution);

                for (BoidIndex i = gridCellStartIndices[currentCellIndex];
                    i <= gridCellEndIndices[currentCellIndex];
                    i++) {

                    if (i == thisBoidIndex) {
                        continue;
                    }

                    // Calculate the values using the boid neighbors for
                    // all three rules at the time.

                    // Rule 1
                    if (glm::distance(pos[thisBoidIndex], pos[i]) < rule1Distance) {
                        perceivedCenterR1 += pos[i];
                        numNeighborsR1++;
                    }

                    // Rule 2
                    if (glm::distance(pos[thisBoidIndex], pos[i]) < rule2Distance) {
                        cR2 -= (pos[i] - pos[thisBoidIndex]);
                    }

                    // Rule 3
                    if (glm::distance(pos[thisBoidIndex], pos[i]) < rule3Distance) {
                        perceivedVelocityR3 += vel1[i];
                        numNeighborsR3++;
                    }

                }
            }
        }
    }


    if (numNeighborsR1 > 0) {
        perceivedCenterR1 /= numNeighborsR1;
    }

    if (numNeighborsR3 > 0) {
        perceivedVelocityR3 /= numNeighborsR3;
    }

    glm::vec3 velChange = ((perceivedCenterR1 - pos[thisBoidIndex]) * rule1Scale)
        + (cR2 * rule2Scale)
        + (perceivedVelocityR3 * rule3Scale);

    glm::vec3 resultVel = vel1[thisBoidIndex] + velChange;

    // Clamp the speed
    resultVel = clampSpeed(resultVel);

    // Record the new velocity into vel2.
    vel2[thisBoidIndex] = resultVel;
    }


/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.


    kernUpdateVelocityBruteForce <<<numBlocks, blockSize>>> (numObjects, dev_pos, dev_vel1, dev_vel2);
    kernUpdatePos <<<numBlocks, blockSize>>> (numObjects, dt, dev_pos, dev_vel1);

  // TODO-1.2 ping-pong the velocity buffers
    glm::vec3* tempPtr = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = tempPtr;
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

   kernComputeIndices << <numBlocks, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, dev_pos,
        dev_particleArrayIndices, dev_particleGridIndices);

    // Wrap device vectors in thrust iterators for use with thrust.
    thrust::device_ptr<BoidIndex> thrust_particleArrayIndices(dev_particleArrayIndices);
    thrust::device_ptr<CellIndex> thrust_particleGridIndices(dev_particleGridIndices);

    thrust::sort_by_key(thrust_particleGridIndices,
                        thrust_particleGridIndices + numObjects,
                        thrust_particleArrayIndices);

    // Let -1 = no pointer / index
    kernResetIntBuffer<<<numBlocks, blockSize>>>(gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <numBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <numBlocks, blockSize >> > (numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    kernUpdateVelNeighborSearchScattered << <numBlocks, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

    kernUpdatePos << <numBlocks, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);

    glm::vec3* tempPtr = dev_vel1;
    dev_vel1 = dev_vel2;
    dev_vel2 = tempPtr;

}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

    kernComputeIndices << <numBlocks, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, dev_pos,
        dev_particleArrayIndices, dev_particleGridIndices);


    // Wrap device vectors in thrust iterators for use with thrust.
    thrust::device_ptr<BoidIndex> thrust_particleArrayIndices(dev_particleArrayIndices);
    thrust::device_ptr<CellIndex> thrust_particleGridIndices(dev_particleGridIndices);

    thrust::sort_by_key(thrust_particleGridIndices,
        thrust_particleGridIndices + numObjects,
        thrust_particleArrayIndices);

    // Let -1 = no pointer / index
    kernResetIntBuffer << <numBlocks, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <numBlocks, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);

    kernIdentifyCellStartEnd << <numBlocks, blockSize >> > (numObjects, dev_particleGridIndices,
        dev_gridCellStartIndices, dev_gridCellEndIndices);

    kernShufflePosVelBuffers << <numBlocks, blockSize >> > (numObjects, dev_particleArrayIndices,
        dev_pos, dev_pos2, dev_vel1, dev_vel2);
    
    kernUpdateVelNeighborSearchCoherent<< <numBlocks, blockSize >> > (numObjects, gridSideCount,
        gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_pos2, dev_vel2, dev_vel1);

    kernUpdatePos << <numBlocks, blockSize >> > (numObjects, dt, dev_pos2, dev_vel2);

    glm::vec3* tempPtrPos = dev_pos;
    dev_pos = dev_pos2;
    dev_pos2 = tempPtrPos;
    
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.

  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  cudaFree(dev_pos2);
}

//#define TEST_THRUST_SORT
//#define TEST_GRID_BOUNDS
//#define TEST_UNIFORM_GRID

std::ostream& operator<<(std::ostream& o, glm::vec3 vec) {
    return o << " [ " << vec.x << ", " << vec.y << ", " << vec.z << " ]";
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

#ifdef TEST_THRUST_SORT
  // test unstable sort
  int *dev_intKeys;
  int* dev_intKeysCopy;
  int *dev_intValues;
  int* dev_intValues2;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  std::unique_ptr<int[]>intKeysCopy{ new int[N] };
  std::unique_ptr<int[]>intValues2{ new int[N] };

  intKeys[0] = intKeysCopy[0] = 0; intValues[0] = 0;
  intKeys[1] = intKeysCopy[1] = 1; intValues[1] = 1;
  intKeys[2] = intKeysCopy[2] = 0; intValues[2] = 2;
  intKeys[3] = intKeysCopy[3] = 3; intValues[3] = 3;
  intKeys[4] = intKeysCopy[4] = 0; intValues[4] = 4;
  intKeys[5] = intKeysCopy[5] = 2; intValues[5] = 5;
  intKeys[6] = intKeysCopy[6] = 2; intValues[6] = 6;
  intKeys[7] = intKeysCopy[7] = 0; intValues[7] = 7;
  intKeys[8] = intKeysCopy[8] = 5; intValues[8] = 8;
  intKeys[9] = intKeysCopy[9] = 6; intValues[9] = 9;

  for (int i = 0; i < 10; i++) {
      intValues2[i] = 10 - i;
  }

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intKeysCopy, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  cudaMalloc((void**)&dev_intValues2, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues2 failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  for (int i = 0; i < N; i++) {
      std::cout << "  key copy: " << intKeysCopy[i];
      std::cout << " value two: " << intValues2[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_keysCopy(dev_intKeysCopy);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  std::cout << "trying to copy, see what happens: " << std::endl;

  //thrust::copy(dev_thrust_keys.begin(), dev_thrust_keys.end(), dev_thrust_intKeysCopy.begin());
  cudaMemcpy(dev_intKeysCopy, dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToDevice);
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intKeysCopy.get(), dev_intKeysCopy, sizeof(int) * N, cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
      std::cout << "  key: " << intKeys[i];
      std::cout << " keyCopy: " << intKeysCopy[i] << std::endl;
  }


  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intKeysCopy);
  cudaFree(dev_intValues);
  cudaFree(dev_intValues2);
  checkCUDAErrorWithLine("cudaFree failed!");

#endif

  /*********************
   VERIFY BOUNDS OF GRID 
   *********************/

  #ifdef TEST_GRID_BOUNDS
  std::cout << "Grid side count (# cells per side): " << gridSideCount << std::endl;
  int x = gridSideCount - 1,
      y = gridSideCount - 1,
      z = gridSideCount - 1;
  int maxIndex = x + y * gridSideCount + z * gridSideCount * gridSideCount;
  std::cout << "Max possible index: " << maxIndex << std::endl;
  std::cout << "Grid cell count (should be maxIndex + 1): " << gridCellCount << std::endl;
  #endif

  /*********************
    UNIT TEST FOR UNIFORM GRID
   *********************/

#ifdef TEST_UNIFORM_GRID
   // Set arbitrary grid parameters
  float test_gridCellWidth = 2.5f;
  int test_gridSideCount = 2;

  int test_gridCellCount = test_gridSideCount * test_gridSideCount * test_gridSideCount;
  float test_gridInverseCellWidth = 1.0f / test_gridCellWidth;
  float test_halfGridWidth = test_gridCellWidth * (test_gridSideCount / 2);
  glm::vec3 test_gridMinimum;
  test_gridMinimum.x -= test_halfGridWidth;
  test_gridMinimum.y -= test_halfGridWidth;
  test_gridMinimum.z -= test_halfGridWidth;

  // Confirm grid range
  std::cout << "Test grid side count (# cells per side): " << test_gridSideCount << std::endl;
  int test_x = test_gridSideCount - 1,
      test_y = test_gridSideCount - 1,
      test_z = test_gridSideCount - 1;
  int test_maxIndex = test_x + test_y * test_gridSideCount + test_z * test_gridSideCount * test_gridSideCount;
  std::cout << "Test gridMin: " << test_gridMinimum.x << ", " << test_gridMinimum.y << ", " << test_gridMinimum.z << std::endl;
  std::cout << "Test max possible index: " << test_maxIndex << std::endl;
  std::cout << "Test grid cell count (should be maxIndex + 1): " << test_gridCellCount << std::endl;

  const int testNumObjects = 4;
  
  glm::vec3* test_pos = new glm::vec3[testNumObjects];
  glm::vec3* test_vel1 = new glm::vec3[testNumObjects];
  glm::vec3* test_vel2 = new glm::vec3[testNumObjects];
  glm::vec3* gpu_pos;
  glm::vec3* gpu_vel1;
  glm::vec3* gpu_vel2;

  BoidIndex* test_particleArrayIndices = new BoidIndex[testNumObjects]; 
  CellIndex* test_particleGridIndices = new CellIndex[testNumObjects];

  BoidIndex* gpu_particleArrayIndices; 
  CellIndex* gpu_particleGridIndices; 

  int* test_gridCellStartIndices = new int[test_gridCellCount]; // What part of dev_particleArrayIndices belongs
  int* test_gridCellEndIndices = new int[test_gridCellCount];   // to this cell?

  int* gpu_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
  int* gpu_gridCellEndIndices;   // to this cell?


  // Allocate buffers
  cudaMalloc((void**)&gpu_pos, testNumObjects * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc test_pos failed!");

  cudaMalloc((void**)&gpu_vel1, testNumObjects * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc test_vel1 failed!");

  cudaMalloc((void**)&gpu_vel2, testNumObjects * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc test_vel2 failed!");

  cudaMalloc((void**)&gpu_particleArrayIndices, testNumObjects * sizeof(BoidIndex));
  checkCUDAErrorWithLine("cudaMalloc test_particleArrayIndices failed!");

  cudaMalloc((void**)&gpu_particleGridIndices, testNumObjects * sizeof(CellIndex));
  checkCUDAErrorWithLine("cudaMalloc test_particleGridIndices failed!");

  cudaMalloc((void**)&gpu_gridCellStartIndices, test_gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc test_gridCellStartIndices failed!");

  cudaMalloc((void**)&gpu_gridCellEndIndices, test_gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc test_gridCellEndIndices failed!");
  
  // Manually set pos and vel values.

  // FIRST CHECK FOR CORNER CASE. Two in same cell,
  // one in different nearby cell, one out of reach

  // Chosen for corner case, center of edge case,
  // center of plane case, center of cube case
  // (first three are clamped and have less cubes to check)

  test_pos[0] = test_gridMinimum + glm::vec3(0.5f);
  test_pos[1] = glm::vec3(1.0f, 1.0f, 1.0f);
  test_pos[2] = test_gridMinimum + glm::vec3(0.5f) + glm::vec3(2.5f, 0.0f, 0.0f);
  test_pos[3] = test_gridMinimum + glm::vec3(1.0f);

  std::cout << "Starting positions: " << std::endl 
      << test_pos[0] << ", " << std::endl
      << test_pos[1] << ", " << std::endl
      << test_pos[2] << ", " << std::endl
      << test_pos[3] << std::endl;

  cudaMemcpy(gpu_pos, test_pos, sizeof(glm::vec3)* testNumObjects, cudaMemcpyHostToDevice);
  
  // Deconstructed version of stepSimulationScatteredGrid(0.20f);
  // TEST COMPUTE INDICES

  kernComputeIndices << <1, blockSize >> > (testNumObjects, test_gridSideCount,
      test_gridMinimum, test_gridInverseCellWidth, gpu_pos,
      gpu_particleArrayIndices, gpu_particleGridIndices);


  cudaMemcpy(test_pos, gpu_pos, sizeof(glm::vec3)* testNumObjects, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_particleArrayIndices, gpu_particleArrayIndices, sizeof(int)* testNumObjects, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_particleGridIndices, gpu_particleGridIndices, sizeof(int)* testNumObjects, cudaMemcpyDeviceToHost);

  std::cout << "Resulting positions: (SHOULD BE SAME)" << std::endl
      << test_pos[0] << ", " << std::endl
      << test_pos[1] << ", " << std::endl
      << test_pos[2] << ", " << std::endl
      << test_pos[3] << std::endl;

  std::wcout << "Array indices: (should be in order)" << std::endl
      << test_particleArrayIndices[0] << ", " << std::endl
      << test_particleArrayIndices[1] << ", " << std::endl
      << test_particleArrayIndices[2] << ", " << std::endl
      << test_particleArrayIndices[3] << std::endl;
  std::cout << "Grid indices:" << std::endl
      << test_particleGridIndices[0] << ", " << std::endl
      << test_particleGridIndices[1] << ", " << std::endl
      << test_particleGridIndices[2] << ", " << std::endl
      << test_particleGridIndices[3] << std::endl;

  // Thrust sort
  

  // needed for use with thrust
  thrust::device_ptr<BoidIndex> test_thrust_particleArrayIndices(gpu_particleArrayIndices);
  thrust::device_ptr<CellIndex> test_thrust_particleGridIndices(gpu_particleGridIndices);

  thrust::sort_by_key(test_thrust_particleGridIndices,
      test_thrust_particleGridIndices + testNumObjects,
      test_thrust_particleArrayIndices);
     
  cudaMemcpy(test_particleArrayIndices, gpu_particleArrayIndices, sizeof(int)* testNumObjects, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_particleGridIndices, gpu_particleGridIndices, sizeof(int)* testNumObjects, cudaMemcpyDeviceToHost);

  std::cout << "Array indices: (should be changed) " << std::endl
      << test_particleArrayIndices[0] << ", " << std::endl
      << test_particleArrayIndices[1] << ", " << std::endl
      << test_particleArrayIndices[2] << ", " << std::endl
      << test_particleArrayIndices[3] << std::endl;
  std::cout << "Grid indices: (should be in order)" << std::endl
      << test_particleGridIndices[0] << ", " << std::endl
      << test_particleGridIndices[1] << ", " << std::endl
      << test_particleGridIndices[2] << ", " << std::endl
      << test_particleGridIndices[3] << std::endl;

  
  // Let -1 = no pointer / index
  kernResetIntBuffer << <1, blockSize >> > (test_gridCellCount, gpu_gridCellStartIndices, -1);
  kernResetIntBuffer << <1, blockSize >> > (test_gridCellCount, gpu_gridCellEndIndices, -1);

  cudaMemcpy(test_gridCellStartIndices, gpu_gridCellStartIndices, sizeof(int)* test_gridCellCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_gridCellEndIndices, gpu_gridCellEndIndices, sizeof(int)* test_gridCellCount, cudaMemcpyDeviceToHost);
 
  std::cout << "Grid Start: (should all be -1)" << std::endl
      << test_gridCellStartIndices[0] << ", " << std::endl
      << test_gridCellStartIndices[1] << ", " << std::endl
      << test_gridCellStartIndices[2] << ", " << std::endl
      << test_gridCellStartIndices[3] << ", " << std::endl
      << test_gridCellStartIndices[4] << ", " << std::endl
      << test_gridCellStartIndices[5] << ", " << std::endl
      << test_gridCellStartIndices[6] << ", " << std::endl
      << test_gridCellStartIndices[7] << std::endl;

  std::cout << "Grid End: (should all be -1)" << std::endl
      << test_gridCellEndIndices[0] << ", " << std::endl
      << test_gridCellEndIndices[1] << ", " << std::endl
      << test_gridCellEndIndices[2] << ", " << std::endl
      << test_gridCellEndIndices[3] << ", " << std::endl
      << test_gridCellEndIndices[4] << ", " << std::endl
      << test_gridCellEndIndices[5] << ", " << std::endl
      << test_gridCellEndIndices[6] << ", " << std::endl
      << test_gridCellEndIndices[7] << std::endl;


  kernIdentifyCellStartEnd << <1, blockSize >> > (testNumObjects, gpu_particleGridIndices,
      gpu_gridCellStartIndices, gpu_gridCellEndIndices);

  cudaMemcpy(test_gridCellStartIndices, gpu_gridCellStartIndices, sizeof(int)* test_gridCellCount, cudaMemcpyDeviceToHost);
  cudaMemcpy(test_gridCellEndIndices, gpu_gridCellEndIndices, sizeof(int)* test_gridCellCount, cudaMemcpyDeviceToHost);

  std::cout << "Grid Start:" << std::endl
      << test_gridCellStartIndices[0] << ", " << std::endl
      << test_gridCellStartIndices[1] << ", " << std::endl
      << test_gridCellStartIndices[2] << ", " << std::endl
      << test_gridCellStartIndices[3] << ", " << std::endl
      << test_gridCellStartIndices[4] << ", " << std::endl
      << test_gridCellStartIndices[5] << ", " << std::endl
      << test_gridCellStartIndices[6] << ", " << std::endl
      << test_gridCellStartIndices[7] << std::endl;

  std::cout << "Grid End:" << std::endl
      << test_gridCellEndIndices[0] << ", " << std::endl
      << test_gridCellEndIndices[1] << ", " << std::endl
      << test_gridCellEndIndices[2] << ", " << std::endl
      << test_gridCellEndIndices[3] << ", " << std::endl
      << test_gridCellEndIndices[4] << ", " << std::endl
      << test_gridCellEndIndices[5] << ", " << std::endl
      << test_gridCellEndIndices[6] << ", " << std::endl
      << test_gridCellEndIndices[7] << std::endl;

  delete[] test_pos;
  delete[] test_vel1;
  delete[] test_vel2;

  delete[] test_particleArrayIndices;
  delete[] test_particleGridIndices;

  cudaFree(test_pos);
  cudaFree(test_vel1);
  cudaFree(test_vel2);
  cudaFree(test_particleArrayIndices);
  cudaFree(test_particleGridIndices);
  cudaFree(test_gridCellStartIndices);
  cudaFree(test_gridCellEndIndices);
#endif

  return;
}
