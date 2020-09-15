**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Qiaosen Chen
  * [LinkedIn](https://www.linkedin.com/in/qiaosen-chen-725699141/), etc.
* Tested on: Windows 10, i5-9400 @ 2.90GHz 16GB, GeForce RTX 2060 6GB (personal computer).



## Part 1: Naive Boids Simulation

![Naive Boids Simulation](https://github.com/giaosame/Project1-CUDA-Flocking/blob/master/images/naive.gif)

## Part 2: Let there be (better) flocking!

- Scattered Grid Simulation

  ![Scattered Grid Simulation](https://github.com/giaosame/Project1-CUDA-Flocking/blob/master/images/scatteredGrid.gif)

- Coherent Grid Simulation

  ![Coherent Grid Simulation](https://github.com/giaosame/Project1-CUDA-Flocking/blob/master/images/coherentGrid.gif)

## Part 3: Performance Analysis

- For each implementation, how does changing the number of boids affect performance? Why do you think this is?

  ![](https://github.com/giaosame/Project1-CUDA-Flocking/blob/master/images/Picture1.jpg)

  Apparently, the performance of the program becomes worse as the number of boids increases, for all 3 simulation methods. Because more boids means more computation, so the frames per second (fps) decreases as the number of boids increases. Also, from the picture above,  we can also know that, the coherent grid simulation does the best performance, the scattered grid simulation ranks the second, and the naive boids simulation is the worst among them. 

  

- For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

  ![](https://github.com/giaosame/Project1-CUDA-Flocking/blob/master/images/Picture2.jpg)

  When the number of boids is 10000, I got the above result. As we can see, the performance almost doesn't be affected as the block size changes. Because each stream multiprocessor can deal with 1 warp each time, while the number of threads of a warp have no relationship with the block size, so the performance is almost the same for different block sizes for the same simulation method.

  

- For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

  ![](https://github.com/giaosame/Project1-CUDA-Flocking/blob/master/images/Picture3.jpg)

  As the grid width increase, it means more coherent uniform grids the simulation algorithm need to deal with, the performance gets worse. I think this result is due to more grids need to be sorted each time.

  

- Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

  ![](https://github.com/giaosame/Project1-CUDA-Flocking/blob/master/images/Picture4.jpg)

  According to the above result, the performance of iterating 27 neighborhood cells is almost the same as the performance of iterating 8 neighborhood cells. 