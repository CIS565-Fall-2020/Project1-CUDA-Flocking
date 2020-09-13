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
// #define blockSize 128
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
glm::vec3* dev_reshuffle_pos;
glm::vec3* dev_reshuffle_vel1;
glm::vec3* dev_reshuffle_vel2;

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
  // 2.2:
  // gridCellWidth = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  // Buffer containing a pointer for each boid to its data in dev_pos and dev_vel1 and dev_vel2
  // Table 2 second row:
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  // Buffer containing the grid index of each boid
  // Table 2 first row:
  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  // Table 1:
  // Buffer containing a pointer for each cell to the beginning of its data in dev_particleArrayIndices
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // TODO-2.3 additional buffers:
  cudaMalloc((void**)&dev_reshuffle_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_reshuffle_pos failed!");

  cudaMalloc((void**)&dev_reshuffle_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_reshuffle_vel1 failed!");

  cudaMalloc((void**)&dev_reshuffle_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_reshuffle_vel2 failed!");
  
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
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
    glm::vec3 temp_self_pos = pos[iSelf];
    glm::vec3 perceived_center(0.0, 0.0, 0.0);
    glm::vec3 rule2_vel(0.0, 0.0, 0.0);
    glm::vec3 rule3_vel(0.0, 0.0, 0.0);
    glm::vec3 res_vel_change(0.0, 0.0, 0.0);
    int rule1_neighbor_count = 0;
    int rule3_neighbor_count = 0;

    for (int iTar = 0; iTar < N; ++iTar)
    {
        if (iTar == iSelf) continue;
        glm::vec3 temp_tar_pos = pos[iTar];
        glm::vec3 temp_tar_vel = vel[iTar];
        float distance = (float) glm::distance(temp_self_pos, temp_tar_pos);
        // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves.
        if (distance < rule1Distance) {
            perceived_center += temp_tar_pos;
            rule1_neighbor_count ++;
        }
        // Rule 2: boids try to stay a distance d away from each other.
        if (distance < rule2Distance) {
            rule2_vel -= (temp_tar_pos - temp_self_pos);
        }
        // Rule 3: boids try to match the speed of surrounding boids.
        if (distance < rule3Distance) {
            rule3_vel += temp_tar_vel;
            rule3_neighbor_count ++;
        }
    }
    // Rule 1 result:
    if (rule1_neighbor_count > 0) {
        perceived_center /= rule1_neighbor_count;
        glm::vec3 rule1_vel = (perceived_center - temp_self_pos) * rule1Scale;
        res_vel_change += rule1_vel;
    }
    // Rule 2 result:
    res_vel_change += (rule2_vel * rule2Scale);
    // Rule 3 result:
    if (rule3_neighbor_count > 0) {
        rule3_vel /= rule3_neighbor_count;
        res_vel_change += (rule3_vel * rule3Scale);
    } 
    
    return res_vel_change;
}



/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
  // Compute a new velocity based on pos and vel1
    glm::vec3 vel_change = computeVelocityChange(N, index, pos, vel1);
    vel2[index] += vel_change;
  // Clamp the speed
    // Record the new velocity into vel2. Question: why NOT vel1?
    if (glm::length(vel2[index]) > maxSpeed) {
        vel2[index] = maxSpeed * glm::normalize(vel2[index]);
    }
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
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
    int particle_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (particle_index >= N) {
        return;
    }
    indices[particle_index] = particle_index;
    glm::vec3 particle_relative_pos = pos[particle_index] - gridMin;
    int grid_x_idx = glm::floor(particle_relative_pos[0] * inverseCellWidth);
    int grid_y_idx = glm::floor(particle_relative_pos[1] * inverseCellWidth);
    int grid_z_idx = glm::floor(particle_relative_pos[2] * inverseCellWidth);
    gridIndices[particle_index] = gridIndex3Dto1D(grid_x_idx, grid_y_idx, grid_z_idx, gridResolution);
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
    // NOTE: N is the number of particles.
    int table_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (table_index >= N) {
        return;
    }
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
    int curr_grid_idx = particleGridIndices[table_index];
    // First table element:
    if (table_index == 0) {
        gridCellStartIndices[curr_grid_idx] = 0;
    }
    // Last table element:
    if (table_index == N - 1) {
        gridCellEndIndices[curr_grid_idx] = N - 1;
    }
    // Look at left:
    if (table_index != 0) {
        int left_grid_index = particleGridIndices[table_index - 1];
        if (left_grid_index != curr_grid_idx) {
            gridCellStartIndices[curr_grid_idx] = table_index;
        }
    }
    // Look at right:
    if (table_index != N - 1) {
        int right_grid_index = particleGridIndices[table_index + 1];
        if (right_grid_index != curr_grid_idx) {
            gridCellEndIndices[curr_grid_idx] = table_index;
        }
    }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
    int particle_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (particle_index >= N) {
        return;
    }
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
    glm::vec3 particle_relative_pos = pos[particle_index] - gridMin;

    glm::vec3 temp_self_pos = pos[particle_index];
    glm::vec3 perceived_center(0.0, 0.0, 0.0);
    glm::vec3 rule2_vel(0.0, 0.0, 0.0);
    glm::vec3 rule3_vel(0.0, 0.0, 0.0);
    glm::vec3 res_vel_change(0.0, 0.0, 0.0);
    int rule1_neighbor_count = 0;
    int rule3_neighbor_count = 0;

    // - Identify which cells may contain neighbors. This isn't always 8. --- Grid width : distance = 2 : 1.
    glm::vec3 round_cell_relative_pos = glm::round(particle_relative_pos * inverseCellWidth);
    for (int i = -1; i < 1; ++i) {
        for (int j = -1; j < 1; ++j) {
            for (int k = -1; k < 1; ++k) {
                int neighbor_x_idx = i + (int)round_cell_relative_pos[0];
                int neighbor_y_idx = j + (int)round_cell_relative_pos[1];
                int neighbor_z_idx = k + (int)round_cell_relative_pos[2];
                // This neighbor cell is out of boundary:
                if (neighbor_x_idx < 0 || neighbor_y_idx < 0 || neighbor_z_idx < 0 || neighbor_x_idx >= gridResolution || neighbor_y_idx >= gridResolution || neighbor_z_idx >= gridResolution) {
                    continue;
                }
                // Get the existing neighbor cell:
                int curr_neighbor_idx = gridIndex3Dto1D(neighbor_x_idx, neighbor_y_idx, neighbor_z_idx, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                int table_start_idx = gridCellStartIndices[curr_neighbor_idx];
                int table_end_idx = gridCellEndIndices[curr_neighbor_idx];
                if (table_start_idx < 0 || table_end_idx < 0) {
                    // This cell doesn't enclose a particle
                    continue;
                }
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                for (int iTar = table_start_idx; iTar <= table_end_idx; ++iTar)
                {
                    int neighbor_particle_idx = particleArrayIndices[iTar];
                    if (neighbor_particle_idx == particle_index) continue;
                    glm::vec3 temp_tar_pos = pos[neighbor_particle_idx];
                    glm::vec3 temp_tar_vel = vel1[neighbor_particle_idx];
                    float distance = (float)glm::distance(temp_self_pos, temp_tar_pos);
                    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves.
                    if (distance < rule1Distance) {
                        perceived_center += temp_tar_pos;
                        rule1_neighbor_count++;
                    }
                    // Rule 2: boids try to stay a distance d away from each other.
                    if (distance < rule2Distance) {
                        rule2_vel -= (temp_tar_pos - temp_self_pos);
                    }
                    // Rule 3: boids try to match the speed of surrounding boids.
                    if (distance < rule3Distance) {
                        rule3_vel += temp_tar_vel;
                        rule3_neighbor_count++;
                    }
                }
            }
        }
    }
    

    // Extra Credit -- Grid Optimization:
    /*
    float maxDistance = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
    glm::vec3 radius_vector(maxDistance, maxDistance, maxDistance);
    glm::vec3 upper_particle_search = temp_self_pos + radius_vector;
    glm::vec3 lower_particle_search = temp_self_pos - radius_vector;
    glm::ivec3 upper_cell_idx = glm::floor((upper_particle_search - gridMin) * inverseCellWidth);
    glm::ivec3 lower_cell_idx = glm::floor((lower_particle_search - gridMin) * inverseCellWidth);
    for (int neighbor_z_idx = lower_cell_idx[2]; neighbor_z_idx <= upper_cell_idx[2]; ++neighbor_z_idx) {
        for (int neighbor_y_idx = lower_cell_idx[1]; neighbor_y_idx <= upper_cell_idx[1]; ++neighbor_y_idx) {
            for (int neighbor_x_idx = lower_cell_idx[0]; neighbor_x_idx <= upper_cell_idx[0]; ++neighbor_x_idx) {
                // This neighbor cell is out of boundary:
                if (neighbor_x_idx < 0 || neighbor_y_idx < 0 || neighbor_z_idx < 0 || neighbor_x_idx >= gridResolution || neighbor_y_idx >= gridResolution || neighbor_z_idx >= gridResolution) {
                    continue;
                }
                // Get the existing neighbor cell:
                int curr_neighbor_idx = gridIndex3Dto1D(neighbor_x_idx, neighbor_y_idx, neighbor_z_idx, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                int table_start_idx = gridCellStartIndices[curr_neighbor_idx];
                int table_end_idx = gridCellEndIndices[curr_neighbor_idx];
                if (table_start_idx < 0 || table_end_idx < 0) {
                    // This cell doesn't enclose a particle
                    continue;
                }
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                for (int iTar = table_start_idx; iTar <= table_end_idx; ++iTar)
                {
                    int neighbor_particle_idx = particleArrayIndices[iTar];
                    if (neighbor_particle_idx == particle_index) continue;
                    glm::vec3 temp_tar_pos = pos[neighbor_particle_idx];
                    glm::vec3 temp_tar_vel = vel1[neighbor_particle_idx];
                    float distance = (float)glm::distance(temp_self_pos, temp_tar_pos);
                    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves.
                    if (distance < rule1Distance) {
                        perceived_center += temp_tar_pos;
                        rule1_neighbor_count++;
                    }
                    // Rule 2: boids try to stay a distance d away from each other.
                    if (distance < rule2Distance) {
                        rule2_vel -= (temp_tar_pos - temp_self_pos);
                    }
                    // Rule 3: boids try to match the speed of surrounding boids.
                    if (distance < rule3Distance) {
                        rule3_vel += temp_tar_vel;
                        rule3_neighbor_count++;
                    }
                }
            }
        }
    }
    */

    /*
    // TODO-2.2: Identify which cells may contain neighbors. This isn't always 27. --- Grid width : distance = 1 : 1.
    int grid_x_idx = glm::floor(particle_relative_pos[0] * inverseCellWidth);
    int grid_y_idx = glm::floor(particle_relative_pos[1] * inverseCellWidth);
    int grid_z_idx = glm::floor(particle_relative_pos[2] * inverseCellWidth);
    glm::vec3 self_cell_idx(grid_x_idx, grid_y_idx, grid_z_idx);
    for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
            for (int k = -1; k < 2; ++k) {
                int neighbor_x_idx = i + (int)self_cell_idx[0];
                int neighbor_y_idx = j + (int)self_cell_idx[1];
                int neighbor_z_idx = k + (int)self_cell_idx[2];
                // This neighbor cell is out of boundary:
                if (neighbor_x_idx < 0 || neighbor_y_idx < 0 || neighbor_z_idx < 0 || neighbor_x_idx >= gridResolution || neighbor_y_idx >= gridResolution || neighbor_z_idx >= gridResolution) {
                    continue;
                }
                // Get the existing neighbor cell:
                int curr_neighbor_idx = gridIndex3Dto1D(neighbor_x_idx, neighbor_y_idx, neighbor_z_idx, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                int table_start_idx = gridCellStartIndices[curr_neighbor_idx];
                int table_end_idx = gridCellEndIndices[curr_neighbor_idx];
                if (table_start_idx < 0 || table_end_idx < 0) {
                    // This cell doesn't enclose a particle
                    continue;
                }
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                for (int iTar = table_start_idx; iTar <= table_end_idx; ++iTar)
                {
                    int neighbor_particle_idx = particleArrayIndices[iTar];
                    if (neighbor_particle_idx == particle_index) continue;
                    glm::vec3 temp_tar_pos = pos[neighbor_particle_idx];
                    glm::vec3 temp_tar_vel = vel1[neighbor_particle_idx];
                    float distance = (float)glm::distance(temp_self_pos, temp_tar_pos);
                    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves.
                    if (distance < rule1Distance) {
                        perceived_center += temp_tar_pos;
                        rule1_neighbor_count++;
                    }
                    // Rule 2: boids try to stay a distance d away from each other.
                    if (distance < rule2Distance) {
                        rule2_vel -= (temp_tar_pos - temp_self_pos);
                    }
                    // Rule 3: boids try to match the speed of surrounding boids.
                    if (distance < rule3Distance) {
                        rule3_vel += temp_tar_vel;
                        rule3_neighbor_count++;
                    }
                }
            }
        }
    }
    */

    // Rule 1 result:
    if (rule1_neighbor_count > 0) {
        perceived_center /= rule1_neighbor_count;
        glm::vec3 rule1_vel = (perceived_center - temp_self_pos) * rule1Scale;
        res_vel_change += rule1_vel;
    }
    // Rule 2 result:
    res_vel_change += (rule2_vel * rule2Scale);
    // Rule 3 result:
    if (rule3_neighbor_count > 0) {
        rule3_vel /= rule3_neighbor_count;
        res_vel_change += (rule3_vel * rule3Scale);
    }
  // - Clamp the speed change before putting the new speed in vel2
    vel2[particle_index] += res_vel_change;
    if (glm::length(vel2[particle_index]) > maxSpeed) {
        vel2[particle_index] = maxSpeed * glm::normalize(vel2[particle_index]);
    }
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
    int particle_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (particle_index >= N) {
        return;
    }
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
    glm::vec3 particle_relative_pos = pos[particle_index] - gridMin;

    glm::vec3 temp_self_pos = pos[particle_index];
    glm::vec3 perceived_center(0.0, 0.0, 0.0);
    glm::vec3 rule2_vel(0.0, 0.0, 0.0);
    glm::vec3 rule3_vel(0.0, 0.0, 0.0);
    glm::vec3 res_vel_change(0.0, 0.0, 0.0);
    int rule1_neighbor_count = 0;
    int rule3_neighbor_count = 0;

    // - Identify which cells may contain neighbors. This isn't always 8. --- Grid width : distance = 2 : 1.
    
    glm::vec3 round_cell_relative_pos = glm::round(particle_relative_pos * inverseCellWidth);
    for (int i = -1; i < 1; ++i) {
        for (int j = -1; j < 1; ++j) {
            for (int k = -1; k < 1; ++k) {
                int neighbor_x_idx = i + (int)round_cell_relative_pos[0];
                int neighbor_y_idx = j + (int)round_cell_relative_pos[1];
                int neighbor_z_idx = k + (int)round_cell_relative_pos[2];
                // This neighbor cell is out of boundary:
                if (neighbor_x_idx < 0 || neighbor_y_idx < 0 || neighbor_z_idx < 0 || neighbor_x_idx >= gridResolution || neighbor_y_idx >= gridResolution || neighbor_z_idx >= gridResolution) {
                    continue;
                }
                // Get the existing neighbor cell:
                int curr_neighbor_idx = gridIndex3Dto1D(neighbor_x_idx, neighbor_y_idx, neighbor_z_idx, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                //   DIFFERENCE: For best results, consider what order the cells should be
                //   checked in to maximize the memory benefits of reordering the boids data.
                int table_start_idx = gridCellStartIndices[curr_neighbor_idx];
                int table_end_idx = gridCellEndIndices[curr_neighbor_idx];
                if (table_start_idx < 0 || table_end_idx < 0) {
                    // This cell doesn't enclose a particle
                    continue;
                }
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                for (int iTar = table_start_idx; iTar <= table_end_idx; ++iTar)
                {
                    int neighbor_particle_idx = iTar;
                    if (neighbor_particle_idx == particle_index) continue;
                    glm::vec3 temp_tar_pos = pos[neighbor_particle_idx];
                    glm::vec3 temp_tar_vel = vel1[neighbor_particle_idx];
                    float distance = (float)glm::distance(temp_self_pos, temp_tar_pos);
                    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves.
                    if (distance < rule1Distance) {
                        perceived_center += temp_tar_pos;
                        rule1_neighbor_count++;
                    }
                    // Rule 2: boids try to stay a distance d away from each other.
                    if (distance < rule2Distance) {
                        rule2_vel -= (temp_tar_pos - temp_self_pos);
                    }
                    // Rule 3: boids try to match the speed of surrounding boids.
                    if (distance < rule3Distance) {
                        rule3_vel += temp_tar_vel;
                        rule3_neighbor_count++;
                    }
                }
            }
        }
    }
    
    /*
    // Extra Credit -- Grid Optimization:
    float maxDistance = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
    glm::vec3 radius_vector(maxDistance, maxDistance, maxDistance);
    glm::vec3 upper_particle_search = temp_self_pos + radius_vector;
    glm::vec3 lower_particle_search = temp_self_pos - radius_vector;
    glm::ivec3 upper_cell_idx = glm::floor((upper_particle_search - gridMin) * inverseCellWidth);
    glm::ivec3 lower_cell_idx = glm::floor((lower_particle_search - gridMin) * inverseCellWidth);
    for (int neighbor_x_idx = lower_cell_idx[0]; neighbor_x_idx <= upper_cell_idx[0]; ++neighbor_x_idx) {
        for (int neighbor_y_idx = lower_cell_idx[1]; neighbor_y_idx <= upper_cell_idx[1]; ++neighbor_y_idx) {
            for (int neighbor_z_idx = lower_cell_idx[2]; neighbor_z_idx <= upper_cell_idx[2]; ++neighbor_z_idx) {
                // This neighbor cell is out of boundary:
                if (neighbor_x_idx < 0 || neighbor_y_idx < 0 || neighbor_z_idx < 0 || neighbor_x_idx >= gridResolution || neighbor_y_idx >= gridResolution || neighbor_z_idx >= gridResolution) {
                    continue;
                }
                // Get the existing neighbor cell:
                int curr_neighbor_idx = gridIndex3Dto1D(neighbor_x_idx, neighbor_y_idx, neighbor_z_idx, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                int table_start_idx = gridCellStartIndices[curr_neighbor_idx];
                int table_end_idx = gridCellEndIndices[curr_neighbor_idx];
                if (table_start_idx < 0 || table_end_idx < 0) {
                    // This cell doesn't enclose a particle
                    continue;
                }
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                for (int iTar = table_start_idx; iTar <= table_end_idx; ++iTar)
                {
                    int neighbor_particle_idx = iTar;
                    if (neighbor_particle_idx == particle_index) continue;
                    glm::vec3 temp_tar_pos = pos[neighbor_particle_idx];
                    glm::vec3 temp_tar_vel = vel1[neighbor_particle_idx];
                    float distance = (float)glm::distance(temp_self_pos, temp_tar_pos);
                    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves.
                    if (distance < rule1Distance) {
                        perceived_center += temp_tar_pos;
                        rule1_neighbor_count++;
                    }
                    // Rule 2: boids try to stay a distance d away from each other.
                    if (distance < rule2Distance) {
                        rule2_vel -= (temp_tar_pos - temp_self_pos);
                    }
                    // Rule 3: boids try to match the speed of surrounding boids.
                    if (distance < rule3Distance) {
                        rule3_vel += temp_tar_vel;
                        rule3_neighbor_count++;
                    }
                }
            }
        }
    }
    */

    // TODO-2.2: Identify which cells may contain neighbors. This isn't always 27. --- Grid width : distance = 1 : 1.
    /*
    int grid_x_idx = glm::floor(particle_relative_pos[0] * inverseCellWidth);
    int grid_y_idx = glm::floor(particle_relative_pos[1] * inverseCellWidth);
    int grid_z_idx = glm::floor(particle_relative_pos[2] * inverseCellWidth);
    glm::vec3 self_cell_idx(grid_x_idx, grid_y_idx, grid_z_idx);
    for (int i = -1; i < 2; ++i) {
        for (int j = -1; j < 2; ++j) {
            for (int k = -1; k < 2; ++k) {
                int neighbor_x_idx = i + (int)self_cell_idx[0];
                int neighbor_y_idx = j + (int)self_cell_idx[1];
                int neighbor_z_idx = k + (int)self_cell_idx[2];
                // This neighbor cell is out of boundary:
                if (neighbor_x_idx < 0 || neighbor_y_idx < 0 || neighbor_z_idx < 0 || neighbor_x_idx >= gridResolution || neighbor_y_idx >= gridResolution || neighbor_z_idx >= gridResolution) {
                    continue;
                }
                // Get the existing neighbor cell:
                int curr_neighbor_idx = gridIndex3Dto1D(neighbor_x_idx, neighbor_y_idx, neighbor_z_idx, gridResolution);
                // - For each cell, read the start/end indices in the boid pointer array.
                int table_start_idx = gridCellStartIndices[curr_neighbor_idx];
                int table_end_idx = gridCellEndIndices[curr_neighbor_idx];
                if (table_start_idx < 0 || table_end_idx < 0) {
                    // This cell doesn't enclose a particle
                    continue;
                }
                // - Access each boid in the cell and compute velocity change from
                //   the boids rules, if this boid is within the neighborhood distance.
                for (int iTar = table_start_idx; iTar <= table_end_idx; ++iTar)
                {
                    int neighbor_particle_idx = iTar;
                    if (neighbor_particle_idx == particle_index) continue;
                    glm::vec3 temp_tar_pos = pos[neighbor_particle_idx];
                    glm::vec3 temp_tar_vel = vel1[neighbor_particle_idx];
                    float distance = (float)glm::distance(temp_self_pos, temp_tar_pos);
                    // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves.
                    if (distance < rule1Distance) {
                        perceived_center += temp_tar_pos;
                        rule1_neighbor_count++;
                    }
                    // Rule 2: boids try to stay a distance d away from each other.
                    if (distance < rule2Distance) {
                        rule2_vel -= (temp_tar_pos - temp_self_pos);
                    }
                    // Rule 3: boids try to match the speed of surrounding boids.
                    if (distance < rule3Distance) {
                        rule3_vel += temp_tar_vel;
                        rule3_neighbor_count++;
                    }
                }
            }
        }
    }
    */

    // Rule 1 result:
    if (rule1_neighbor_count > 0) {
        perceived_center /= rule1_neighbor_count;
        glm::vec3 rule1_vel = (perceived_center - temp_self_pos) * rule1Scale;
        res_vel_change += rule1_vel;
    }
    // Rule 2 result:
    res_vel_change += (rule2_vel * rule2Scale);
    // Rule 3 result:
    if (rule3_neighbor_count > 0) {
        rule3_vel /= rule3_neighbor_count;
        res_vel_change += (rule3_vel * rule3Scale);
    }
    // - Clamp the speed change before putting the new speed in vel2
    vel2[particle_index] = vel1[particle_index] + res_vel_change;
    if (glm::length(vel2[particle_index]) > maxSpeed) {
        vel2[particle_index] = maxSpeed * glm::normalize(vel2[particle_index]);
    }
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
    // Update velocity:
    kernUpdateVelocityBruteForce <<< fullBlocksPerGrid, blockSize >>> (numObjects, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");
    // Update position:
    kernUpdatePos <<< fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");
    // TODO-1.2 ping-pong the velocity buffers
    dev_vel1 = dev_vel2;
}

__global__ void particle_grid_idx_velocity_test(int N, int* particleGridIndices, glm::vec3* pos, glm::vec3* vel2)
{
    int index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (index >= N) {
        return;
    }
    vel2[index][0] = (float)particleGridIndices[index];
    vel2[index][1] = (float)particleGridIndices[index];
    vel2[index][2] = (float)particleGridIndices[index];
    if (glm::length(vel2[index]) > maxSpeed) {
        vel2[index] = maxSpeed * glm::normalize(vel2[index]);
    }
}

__global__ void sorted_idx_velocity_test(int N, int* particleGridIndices, int* gridCellStartIndices, int* gridCellEndIndices,
    int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel2)
{
    int table_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (table_index >= N) {
        return;
    }
    int curr_particle_index = particleArrayIndices[table_index];
    vel2[curr_particle_index][0] = (float)particleGridIndices[table_index];
    vel2[curr_particle_index][1] = (float)particleGridIndices[table_index];
    vel2[curr_particle_index][2] = (float)particleGridIndices[table_index];
    if (glm::length(vel2[curr_particle_index]) > maxSpeed) {
        vel2[curr_particle_index] = maxSpeed * glm::normalize(vel2[curr_particle_index]);
    }
}

// NOTE: dev_particleGridIndices[p_idx] gives the grid idx of the p_idx particle.
void Boids::stepSimulationScatteredGrid(float dt) {
    
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 cell_block_num((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer <<< cell_block_num, blockSize >>> (gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
    kernResetIntBuffer <<< cell_block_num, blockSize >>> (gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
    kernComputeIndices <<< fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd <<< fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchScattered <<< fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth,
        dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");
    
  // - Update positions
    kernUpdatePos <<< fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");
  // - Ping-pong buffers as needed
    dev_vel1 = dev_vel2;
    
}

__global__ void reshuffle_particle_data(int N, int* particleArrayIndices, glm::vec3* pos, glm::vec3* vel1, glm::vec3* pos_reshuffle, glm::vec3* vel1_reshuffle)
{
    int table_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (table_index >= N) {
        return;
    }
    int particle_index = particleArrayIndices[table_index];
    pos_reshuffle[table_index] = pos[particle_index];
    vel1_reshuffle[table_index] = vel1[particle_index];
}


__global__ void shuffle_back_particle_data(int N, int* particleArrayIndices, glm::vec3* vel2, glm::vec3* vel2_reshuffle)
{
    int table_index = threadIdx.x + (blockIdx.x * blockDim.x);
    if (table_index >= N) {
        return;
    }
    int particle_index = particleArrayIndices[table_index];
    vel2[particle_index] = vel2_reshuffle[table_index];
}

void Boids::stepSimulationCoherentGrid(float dt) {
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
    dim3 cell_block_num((gridCellCount + blockSize - 1) / blockSize);
    kernResetIntBuffer << < cell_block_num, blockSize >> > (gridCellCount, dev_gridCellStartIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
    kernResetIntBuffer << < cell_block_num, blockSize >> > (gridCellCount, dev_gridCellEndIndices, -1);
    checkCUDAErrorWithLine("kernResetIntBuffer failed!");
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
    kernComputeIndices << < fullBlocksPerGrid, blockSize >> > (numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
    checkCUDAErrorWithLine("kernComputeIndices failed!");
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
    thrust::device_ptr<int> dev_thrust_keys(dev_particleGridIndices);
    thrust::device_ptr<int> dev_thrust_values(dev_particleArrayIndices);
    thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + numObjects, dev_thrust_values);
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
    kernIdentifyCellStartEnd << < fullBlocksPerGrid, blockSize >> > (numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
    checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
    reshuffle_particle_data <<< fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleArrayIndices, dev_pos, dev_vel1, dev_reshuffle_pos, dev_reshuffle_vel1);
    checkCUDAErrorWithLine("reshuffle_particle_data failed!");
  // - Perform velocity updates using neighbor search
    kernUpdateVelNeighborSearchCoherent <<< fullBlocksPerGrid, blockSize >>> (numObjects, gridSideCount, gridMinimum, 
        gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_reshuffle_pos, dev_reshuffle_vel1, dev_reshuffle_vel2);
    checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");
    shuffle_back_particle_data <<< fullBlocksPerGrid, blockSize >>> (numObjects, dev_particleArrayIndices, dev_vel2, dev_reshuffle_vel2);
  // - Update positions
    kernUpdatePos <<< fullBlocksPerGrid, blockSize >>> (numObjects, dt, dev_pos, dev_vel2);
    checkCUDAErrorWithLine("kernUpdatePos failed!");
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
    dev_vel1 = dev_vel2;
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
  cudaFree(dev_reshuffle_pos);
  cudaFree(dev_reshuffle_vel1);
  cudaFree(dev_reshuffle_vel2);
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
