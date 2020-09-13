**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Ling Xie
  * [LinkedIn](https://www.linkedin.com/in/ling-xie-94b939182/), 
  * [personal website](https://jack12xl.netlify.app).
* Tested on: 
  * Windows 10, Intel(R) Xeon(R) CPU E5-2650 v4 @ 2.20GHz 2.20GHz ( two processors) 
  * 64.0 GB memory
  * NVIDIA TITAN XP GP102

Thanks to [FLARE LAB](http://faculty.sist.shanghaitech.edu.cn/faculty/liuxp/flare/index.html) for this ferocious monster.

# Intro

In this project, we mainly implement a flocking simulation based on the [Reynolds Boids](http://www.kfish.org/boids/pseudocode.html) algorithm with CUDA. 

Basically, we implement how the boids interact with each other, which is defined in algorithm. The rendering pipeline of openGL is provided as basic code setup in advance.

Here shows the demo. Each primitives' RGB color represents the XYZ  of its speed. 

First the boids are running mindlessly.

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/demo_3.gif)

Then it tends to gradually pull together.
![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/demo_2.gif)




# Task

In this project, we manage to fulfill all the requirements:

1. **Baseline**: Naively traverse all the existing boids
2. **Uniform Grid**: Add spatial uniform grid for searchings.
3. **Coherent**: Based on **uniform grid**, align the velocity and position attributes in memory for large memory bandwidth.

And extra points:

	1. shared memory(not implemented yet)
 	2. grid loop optimization

# Performance analysis

The following experiment runs under with default project configuration.

- Default hyper parameters setting if not mentioned
- Run in total 16192 frames

Since after a few frames, boids are tending to stick together, which could bring substantial pressure on neighbour grid searching(frame rate would drop), we run these [experiments](https://github.com/Jack12xl/Project1-CUDA-Flocking#experiments) under certain frames and get the average FPS for fairness.  

As required, we implement three methods on searching the neighbours.

The following image would give you a rough idea of how much time these three methods in CUDA kernel would consume in one time step:

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/Kernal-Update-Velocity-Camparison.svg)



#### Experiments

As required by [instructions](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/INSTRUCTION.md), we show the experiments results here.

##### 1. Frame rate change with increasing **# of boids** for naive, scattered uniform grid, and coherent uniform grid (render off)

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/performance_boid.svg)

**Observation**: Obviously and intuitively, the frame rate per second drops drastically when the number of boids increase.

[Further discussion here](https://github.com/Jack12xl/Project1-CUDA-Flocking#questions-and-answers)

##### 2. Frame rate change with increasing **block size** for naive, scattered uniform grid, and coherent uniform grid( render off )

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/performance_block.svg)

**Observation**: The performance goes up as the block size changes from 64 to 128. Then it rarely gets increased too much.  

[Further discussion here](https://github.com/Jack12xl/Project1-CUDA-Flocking#questions-and-answers)

# Extra Credit

Here goes the extra points part.



#### Optimization with shared memory

Paper citation: 

#### Grid-Looping Optimization

Actually, our first intuitive idea is exactly to use input given range for searching, instead of hard coding the neighbor cell index, which is somehow fast in certain cases(8 and 27) but not elegant or versatile for configuring.  

Here shows the performance figure under different cell width:

![alt text](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/FPS_performance_cell_size.png)

# Questions and Answers

As required by [instruction](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/INSTRUCTION.md), here we answer these questions respectively:

- For each implementation, how does changing the number of boids affect performance? Why do you think this is?

  - Intuitively and [empirically](), as the number goes up, the fps would drop correspondingly.

    - First, a larger number of boids means more primitives to search and process during applying the three rules.
    -   More primitives leads to more drawing call during rendering.

    

- For each implementation, how does changing the block count and block size affect performance? Why do you think this is?
  - First the performance rises up when block size increases from 64 to 128
    - possibly due to 128 is the critical value to cover **latency issues when accessing registers**
  - Then the performance rises steadily,
    - The latency issue is solved so the performance would not increase too much.
  - reference link, 
    - [stackoverflow](https://stackoverflow.com/questions/9985912/how-do-i-choose-grid-and-block-dimensions-for-cuda-kernels)
    - [nvidia forum](https://forums.developer.nvidia.com/t/how-to-decide-the-optimal-block-size-in-cuda/14906)
- For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?
  - From image link, the advantage of coherent uniform grid increase rapidly as the number of boids increase.
    - We think this is due to as more and more boids are in grids, the advantage of high memory bandwidth from coherency shines brighter. Since the coherent aligns the velocity and positions attribute.
- Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!
  - As is showed in [this image](https://github.com/Jack12xl/Project1-CUDA-Flocking/blob/master/images/FPS_performance_cell_size.png), we keep the search range and vary the cell size, by which we change the number of neighbors during searching. The larger size a cell is, more neighboring cells the algorithm would check.
    - We find that local maximum of performance is reached in cell size is 1 * searchRange (at most 27 neighbors), which in our testing cases 27 neighbors searching is faster than 8 (in diagram 2 * cell size, in other word the default settings).
    - So yes, indeed it will affect performance.
      - Critical thinking plays a great role here and basically:
        - Sparser grids(large cell width ) cannot eliminate as many intersection candidates but a higher resolution(small cell width) might result in bigger cost for traversal.
        - Actually it's a cost trade-off between intersection and traversal.
        - So in our case, traversal cost is less than the benefit of more intersection candidates. 
  - Paper reference: 
    - A Parallel Algorithm for Construction of Uniform Grids
    - Improving Boids Algorithm in GPU using Estimated Self Occlusion