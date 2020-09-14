#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

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
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

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
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3* dev_coherent_pos;
glm::vec3* dev_coherent_vel;


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
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int)); // What index in dev_pos and dev_velX represents this particle?
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int)); // What grid cell is this particle in?
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  // needed for use with thrust
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_pointer_cast<int>(dev_particleGridIndices);

  // What part of dev_particleArrayIndices belongs to this cell?
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int)); // What index in dev_pos and dev_velX represents this particle?
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int)); // What grid cell is this particle in?
  checkCUDAErrorWithLine("cudaMalloc ddev_gridCellEndIndices failed!");

  // for coherent grid part
  cudaMalloc((void**)&dev_coherent_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_coherentPos failed!");

  cudaMalloc((void**)&dev_coherent_vel, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_coherentVel1 failed!");

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
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3* pos, const glm::vec3* vel) {

// Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
// Rule 2: boids try to stay a distance d away from each other    
// Rule 3: boids try to match the speed of surrounding boids

    // need to initialize some variables for the different rules
    // rule1
    glm::vec3 perceived_center(0, 0, 0);
    int rule1_number_of_neighbors = 0;

    // rule2
    glm::vec3 c(0, 0, 0);

    // rule3
    glm::vec3 perceived_velocity(0, 0, 0);
    int rule3_number_of_neighbors = 0;

    for (int i = 0; i < N; i++) {
        if (i != iSelf) {
            float distance = glm::distance(pos[i], pos[iSelf]);

            if (distance < rule1Distance) {
                perceived_center += pos[i];
                rule1_number_of_neighbors++;
            }
            if (distance < rule2Distance) {
                c -= (pos[i] - pos[iSelf]);
            }
            if (distance < rule3Distance) {
                perceived_velocity += vel[i];
                rule3_number_of_neighbors++;
            }

        }
    
    }

    // The influence velocities from rule 1, 2, and 3
    glm::vec3 rule1_velocity(0, 0, 0);
    glm::vec3 rule2_velocity(0, 0, 0);
    glm::vec3 rule3_velocity(0, 0, 0);

    if (rule1_number_of_neighbors > 0) {
        perceived_center /= rule1_number_of_neighbors;
        rule1_velocity = (perceived_center - pos[iSelf]) * rule1Scale;
    }

    rule2_velocity = c * rule2Scale;

    if (rule3_number_of_neighbors > 0) {
        perceived_velocity /= rule3_number_of_neighbors;
        rule3_velocity = perceived_velocity * rule3Scale;
    }

    return rule1_velocity + rule2_velocity + rule3_velocity;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3* pos,
    glm::vec3* vel1, glm::vec3* vel2) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // Compute a new velocity based on pos and vel1
    glm::vec3 new_velocity = computeVelocityChange(N, index, pos, vel1) + vel1[index];

    // clamp the speed so that the magnitude of the velocity vector < maxspeed
    if (glm::length(new_velocity) > maxSpeed) {
        new_velocity = glm::normalize(new_velocity) * maxSpeed;
    }

    // Record the new velocity into vel2
    vel2[index] = new_velocity;
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

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    // - Label each boid with the index of its grid cell.
    int iX = glm::floor((pos[index].x - gridMin.x) * inverseCellWidth);
    int iY = glm::floor((pos[index].y - gridMin.y) * inverseCellWidth);
    int iZ = glm::floor((pos[index].z - gridMin.z) * inverseCellWidth);
    int index1D = gridIndex3Dto1D(iX, iY, iZ, gridResolution);
    gridIndices[index] = index1D;

    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    indices[index] = index;
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
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

    // some cornercases to think of: 
    // 1. All the boids are in one cell 
    // 2. All the boids are in different cells (1 cell per boid)

    // if it's the first one, we want to make sure there's a start
    if (index == 0) {
        gridCellStartIndices[particleGridIndices[index]] = index;
        return;
    }

    // if it's the last one, we want to make sure there's an end
    // here we don't return because we want to make sure that we start/end if last one
    // is different than previous
    if (index == N - 1) {
        gridCellEndIndices[particleGridIndices[index]] = index;
    }

    // everything in between is covered
    if (particleGridIndices[index] != particleGridIndices[index - 1]) {
        gridCellStartIndices[particleGridIndices[index]] = index; 
        gridCellEndIndices[particleGridIndices[index - 1]] = index - 1;
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {

  // - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
    // Note: Remember that this is the array index, NOT the boid index because particleArrayIndices is reordered!
    int array_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (array_index >= N) {
        return;
    }


    // Identify the grid cell that this particle is in
    // calculate the grid cell's x, y, z
    int index = particleArrayIndices[array_index];
    glm::vec3 grid_coords = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    int grid_index1D = gridIndex3Dto1D(grid_coords.x, grid_coords.y, grid_coords.z, gridResolution);

    // find which corner we are close to so that we know which 8 grids to look at
    glm::vec3 side = glm::sign((pos[index] - gridMin) - (grid_coords + glm::vec3(0.5)) * cellWidth);

    // need to initialize some variables for the different rules
    // rule1
    glm::vec3 perceived_center(0, 0, 0);
    int rule1_number_of_neighbors = 0;

    // rule2
    glm::vec3 c(0, 0, 0);

    // rule3
    glm::vec3 perceived_velocity(0, 0, 0);
    int rule3_number_of_neighbors = 0;

  // - Identify which cells may contain neighbors. This isn't always 8.
    // find neighbor cells based on the the sign of the sides
    // the nested floor loop will give us these 8 combinations: 
    // 1. (0, 0, 0)
    // 2. (sidex, 0, 0)
    // 3. (0, sidey, 0)
    // 4. (sidex, sidey, 0)
    // 5. (0, 0, sidez)
    // 6. (sidex, 0, sidez)
    // 7. (0, sidey, sidez)
    // 8. (sidex, sidey, sidez)
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                glm::vec3 neighbor_coords = grid_coords + glm::vec3(x, y, z) * side;

                // need to check if neighbor coords are out of bounds!
                if (neighbor_coords.x < 0 || neighbor_coords.x >= gridResolution ||
                    neighbor_coords.y < 0 || neighbor_coords.y >= gridResolution ||
                    neighbor_coords.z < 0 || neighbor_coords.z >= gridResolution) {
                    continue;
                }

                // need to convert 3d to 1d
                int neighbor_index1d = gridIndex3Dto1D(neighbor_coords.x, neighbor_coords.y, neighbor_coords.z, gridResolution);
            
                // - For each cell, read the start/end indices in the boid pointer array.
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                int start_index = gridCellStartIndices[neighbor_index1d];
                int end_index = gridCellEndIndices[neighbor_index1d];
                // uh oh, somehow the end is less than the start! 
                if (end_index < start_index) {
                    continue;
                }
                // there are no boids in this cell
                if (start_index == -1) {
                    continue;
                }

                for (int i = start_index; i <= end_index; i++) {
                    int neighbor_boid_index = particleArrayIndices[i];
                    if (neighbor_boid_index != index) {
                        float distance = glm::length(pos[index] - pos[neighbor_boid_index]);

                        if (distance < rule1Distance) {
                            perceived_center += pos[neighbor_boid_index];
                            rule1_number_of_neighbors++;
                        }
                        if (distance < rule2Distance) {
                            c -= (pos[neighbor_boid_index] - pos[index]);
                        }
                        if (distance < rule3Distance) {
                            perceived_velocity += vel1[neighbor_boid_index];
                            rule3_number_of_neighbors++;
                        }
                    }
                }
            }
        }
    }
    
    // The influence velocities from rule 1, 2, and 3
    glm::vec3 rule1_velocity(0, 0, 0);
    glm::vec3 rule2_velocity(0, 0, 0);
    glm::vec3 rule3_velocity(0, 0, 0);

    if (rule1_number_of_neighbors > 0) {
        perceived_center /= rule1_number_of_neighbors;
        rule1_velocity = (perceived_center - pos[index]) * rule1Scale;
    }

    rule2_velocity = c * rule2Scale;

    if (rule3_number_of_neighbors > 0) {
        perceived_velocity /= rule3_number_of_neighbors;
        rule3_velocity = perceived_velocity * rule3Scale;
    }

    glm::vec3 new_velocity = vel1[index] + rule1_velocity + rule2_velocity + rule3_velocity;

    // clamp the speed so that the magnitude of the velocity vector < maxspeed
    if (glm::length(new_velocity) > maxSpeed) {
        new_velocity = glm::normalize(new_velocity) * maxSpeed;
    }

    // Record the new velocity into vel2
    vel2[index] = new_velocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.

    // - Update a boid's velocity using the uniform grid to reduce
    // the number of boids that need to be checked
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // Identify the grid cell that this particle is in
    // calculate the grid cell's x, y, z
    // DIFFERENCE: This time, the index directly correlates with the position/velocity vecotr because those are reordered
    glm::vec3 grid_coords = glm::floor((pos[index] - gridMin) * inverseCellWidth);
    int grid_index1D = gridIndex3Dto1D(grid_coords.x, grid_coords.y, grid_coords.z, gridResolution);

    // find which corner we are close to so that we know which 8 grids to look at
    glm::vec3 side = glm::sign((pos[index] - gridMin) - (grid_coords + glm::vec3(0.5)) * cellWidth);

    // need to initialize some variables for the different rules
    // rule1
    glm::vec3 perceived_center(0, 0, 0);
    int rule1_number_of_neighbors = 0;

    // rule2
    glm::vec3 c(0, 0, 0);

    // rule3
    glm::vec3 perceived_velocity(0, 0, 0);
    int rule3_number_of_neighbors = 0;

  // - Identify which cells may contain neighbors. This isn't always 8.
  // find neighbor cells based on the the sign of the sides
  // the nested floor loop will give us these 8 combinations: (in this order) 
  // 1. (0, 0, 0)
  // 2. (sidex, 0, 0)
  // 3. (0, sidey, 0)
  // 4. (sidex, sidey, 0)
  // 5. (0, 0, sidez)
  // 6. (sidex, 0, sidez)
  // 7. (0, sidey, sidez)
  // 8. (sidex, sidey, sidez)
    // this way, we are able to check x first
    for (int z = 0; z < 2; z++) {
        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                glm::vec3 neighbor_coords = grid_coords + glm::vec3(x, y, z) * side;

                // need to check if neighbor coords are out of bounds!
                if (neighbor_coords.x < 0 || neighbor_coords.x >= gridResolution ||
                    neighbor_coords.y < 0 || neighbor_coords.y >= gridResolution ||
                    neighbor_coords.z < 0 || neighbor_coords.z >= gridResolution) {
                    continue;
                }

                // need to convert 3d to 1d
                int neighbor_index1d = gridIndex3Dto1D(neighbor_coords.x, neighbor_coords.y, neighbor_coords.z, gridResolution);

                // - For each cell, read the start/end indices in the boid pointer array.
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                int start_index = gridCellStartIndices[neighbor_index1d];
                int end_index = gridCellEndIndices[neighbor_index1d];
                // uh oh, somehow the end is less than the start! 
                if (end_index < start_index) {
                    continue;
                }
                // there are no boids in this cell
                if (start_index == -1) {
                    continue;
                }

                for (int i = start_index; i <= end_index; i++) {
                    int neighbor_boid_index = i; // DIFFERENCE: We are refering to the pos,vel directly now after rearranging
                    if (neighbor_boid_index != index) {
                        float distance = glm::length(pos[index] - pos[neighbor_boid_index]);

                        if (distance < rule1Distance) {
                            perceived_center += pos[neighbor_boid_index];
                            rule1_number_of_neighbors++;
                        }
                        if (distance < rule2Distance) {
                            c -= (pos[neighbor_boid_index] - pos[index]);
                        }
                        if (distance < rule3Distance) {
                            perceived_velocity += vel1[neighbor_boid_index];
                            rule3_number_of_neighbors++;
                        }
                    }
                }
            }
        }
    }

    // The influence velocities from rule 1, 2, and 3
    glm::vec3 rule1_velocity(0, 0, 0);
    glm::vec3 rule2_velocity(0, 0, 0);
    glm::vec3 rule3_velocity(0, 0, 0);

    if (rule1_number_of_neighbors > 0) {
        perceived_center /= rule1_number_of_neighbors;
        rule1_velocity = (perceived_center - pos[index]) * rule1Scale;
    }

    rule2_velocity = c * rule2Scale;

    if (rule3_number_of_neighbors > 0) {
        perceived_velocity /= rule3_number_of_neighbors;
        rule3_velocity = perceived_velocity * rule3Scale;
    }

    glm::vec3 new_velocity = vel1[index] + rule1_velocity + rule2_velocity + rule3_velocity;

    // clamp the speed so that the magnitude of the velocity vector < maxspeed
    if (glm::length(new_velocity) > maxSpeed) {
        new_velocity = glm::normalize(new_velocity) * maxSpeed;
    }

    // Record the new velocity into vel2
    vel2[index] = new_velocity;
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {

    // use the kernels you wrote to step the simulation forward in time.
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel1);
    kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_pos, dev_vel1, dev_vel2);

    // ping-pong the velocity buffers
    std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize); // we need this so that we can reset the start and end arrays, which have a size = grid cell count
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
    // call kernComputeIndices with the two dev arrays
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    // set up thrust stuff and sort
    // make sure to use gridindices as key and particlearrayindices as values
    dev_thrust_particleArrayIndices = thrust::device_pointer_cast<int>(dev_particleArrayIndices);
    dev_thrust_particleGridIndices = thrust::device_pointer_cast<int>(dev_particleGridIndices);
    thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
     // call kernResetIntBuffer(int N, int *intBuffer, int value) to reset the buffer
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
     // call kernIdentifyCellStartEnd 
    kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices,
        dev_gridCellEndIndices);

  // - Perform velocity updates using neighbor search
    // call kernUpdateVelNeighborSearchScattered
    kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
        gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices,
        dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);

  // - Update positions
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

  // - Ping-pong buffers as needed
    std::swap(dev_vel1, dev_vel2);
}

/*
* Helper function that takes pos and vel and puts the rearranged version into coherent pos/vel
*/
__global__ void kernRearrangePosAndVel(int N, int* particleArrayIndices, glm::vec3* pos,
    glm::vec3* vel1, glm::vec3* coherent_pos, glm::vec3* coherent_vel) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }

    // rearranged pos and vel go into coherent_pos/vel
    int boid_index = particleArrayIndices[index];
    coherent_pos[index] = pos[boid_index];
    coherent_vel[index] = vel1[boid_index];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 fullBlocksPerGridCells((gridCellCount + blockSize - 1) / blockSize);

  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
    // call kernComputeIndices with the two dev arrays
       kernComputeIndices << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, 
           dev_pos, dev_particleArrayIndices, dev_particleGridIndices);

  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    // set up thrust stuff and sort
       dev_thrust_particleArrayIndices = thrust::device_pointer_cast<int>(dev_particleArrayIndices);
       dev_thrust_particleGridIndices = thrust::device_pointer_cast<int>(dev_particleGridIndices);
       thrust::sort_by_key(dev_thrust_particleGridIndices, dev_thrust_particleGridIndices + numObjects, dev_thrust_particleArrayIndices);


  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    // call kernResetIntBuffer(int N, int *intBuffer, int value) to reset the buffer
    // call kernIdentifyCellStartEnd
       kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
       kernResetIntBuffer << <fullBlocksPerGridCells, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
       // call kernIdentifyCellStartEnd 
       kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices,
           dev_gridCellEndIndices);

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
       kernRearrangePosAndVel << <fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_coherent_pos, dev_coherent_vel);
       
  // - Perform velocity updates using neighbor search
       kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth,
           gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_coherent_pos, dev_coherent_vel, dev_vel2);

  // - Update positions
       std::swap(dev_pos, dev_coherent_pos);
       kernUpdatePos << <fullBlocksPerGrid, blockSize >> > (numObjects, dt, dev_pos, dev_vel2);

  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
       std::swap(dev_vel1, dev_vel2);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);

  cudaFree(dev_coherent_pos);
  cudaFree(dev_coherent_vel);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
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

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
