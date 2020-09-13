**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Weiyu Du
  * [LinkedIn](https://www.linkedin.com/in/weiyu-du/)
* Tested on: CETS virtual lab MOR100C-08, Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

### Boids Simulation
<img src="https://github.com/WeiyuDu/Project1-CUDA-Flocking/blob/master/images/boid.png" width="600"/>
<img src="https://github.com/WeiyuDu/Project1-CUDA-Flocking/blob/master/images/demo1.gif" width="600"/>

### Performance Analysis
1) Plot of number of boids versus frame rate without visualization (x axis: 5000, 10000, 20000, 50000)
<img src="https://github.com/WeiyuDu/Project1-CUDA-Flocking/blob/master/images/numboid_novis.png"/>
2) Plot of number of boids versus frame rate with visualization (x axis: 5000, 10000, 20000, 50000)
<img src="https://github.com/WeiyuDu/Project1-CUDA-Flocking/blob/master/images/numboid_vis.png"/>
3) Plot of block size versus frame rate (without visualization, x axis: 128, 256, 512, 1024)
<img src="https://github.com/WeiyuDu/Project1-CUDA-Flocking/blob/master/images/numblock.png"/>

### Questions
1) **For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

Performance decreases as the number of boids increases. This is because more boids require more threads for kernels that require significant operations, for example kernUpdateVelNeighborSearch, kernUpdatePos, kernComputeIndices.

2) **For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

Performance does not change as block size increase. This is because larger block size would only require more threads for kernels that have very simple operation, for example kernResetIntBuffer.

3) **For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

Yes, I experienced performance improvements with coherent uniform grid from scattered grid, but not as significant as from naive to scattered grid. This is the outcome I expect because we cut out the time to access dev_particleArrayIndices memory for each thread, which improves performance.
 
4) **Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not?**

It does not affect performance much. We still have the same number of threads.
