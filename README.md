**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Name: Gizem Dal
  * [LinkedIn](https://www.linkedin.com/in/gizemdal), [personal website](https://www.gizemdal.com/)
* Tested on: Predator G3-571 Intel(R) Core(TM) i7-7700HQ CPU @ 2.80 GHz 2.81 GHz - Personal computer (borrowed my friend's computer for the semester)

## CUDA Flocking

* Boid simulation running in Coherent Uniform Grid mode - 50000 particles (FPS capped to 35):

![Coherent mode gif](images/coherent_50000.gif)

**Questions:**
* For each implementation, how does changing the number of boids affect performance? Why do you think this is?

*Answer:* Increasing the number of boids slows down the performance in all modes (naive, uniform grid, coherent uniform grid) while decreasing it improves performance. In every simulation step, we find the neighbors of every single boid and update velocities and positions. In naive mode, the complexity scales up pretty quickly since our naive method iterates over all the boids to find neighbors for every single boid. We can observe a significant difference between result of increasing boid count in naive mode versus coherent/uniform grid modes since the code is significantly optimized to only iterate over "likely neighboring" boids within a grid range.

* For each implementation, how does changing the block count and block size affect performance? Why do you think this is?

*Answer:* It seems like changing the block count or the block size doesn't have a significant effect on the performance of the simulation (although I was expecting it to have an impact). One possible explanation I could think of is because we're calling the cudaDeviceSynchronize() function inside init simulation, which forces the device to wait until all the computation from previous tasks finish then proceed to the next computation.

* For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?

*Answer:* I experienced about a 20-30 fps improvement on the performance with the coherent uniform grid. I believe the reason for that is because we're reshuffling the position and velocity buffers to have boid data sorted and this allows the device to access this sequential data more efficiently versus accessing data scattered in memory.

* Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!

*Answer:* Besides having to check more cells, we also have to check for more neighbor boids when we decrease our cellWidth. Decreasing the cell width while keeping the search radius constant results in having to account for more neighboring cells, thus we increase the possible number of boids that are within our search radius. If we also increase the number of boids (but keep the scene_scale the same), every cell has a higher chance of enclosing more boids compared to before. All of these conditions can impact the performance when we check 27 vs 8 neighboring cells.

I'm still trying to figure out what is the best way to add graphs to GitHub README files, so for this assignment I will be using tables instead to represent the analysis.

**Timing the kernels**

I set a CUDA event timer inside the step simulation functions for all 3 modes and recorded the elapsed time between before and after the kernels are run to compute the new velocities and update the positions of all the particles. Here are the results I have observed with *N = 10000, blockSize = 128, VISUALIZATION = 1*:

Mode | Naive | Naive Uniform | Semi-Coherent Uniform
:---: | :---: | :---: | :---:
Time (in ms) | 15-16 | 2.7-3.4 | 2.6-3.2

**Impact of particle count on performance**

I recorded the frame rate per second for all 3 simulation modes while maintaining the blockSize equal to 128 but changing the N value. I noted down the approximate fps values on the table below since the value changes very frequently.

* **Visualization enabled, N = 10000**

Mode | Naive | Naive Uniform | Semi-Coherent Uniform
:---: | :---: | :---: | :---:
Fps | 56 | 205 | 225

* **Visualization disabled, N = 10000**

Mode | Naive | Naive Uniform | Semi-Coherent Uniform
:---: | :---: | :---: | :---:
Fps | 60 | 287 | 310

* **Visualization enabled, N = 50000**

Mode | Naive | Naive Uniform | Semi-Coherent Uniform
:---: | :---: | :---: | :---:
Fps | 2.3 | 28.6 | 29.9

* **Visualization disabled, N = 50000**

Mode | Naive | Naive Uniform | Semi-Coherent Uniform
:---: | :---: | :---: | :---:
Fps | 2.3 | 29.6 | 30.9

**Impact of blockSize on performance**

As my answer to the question in the previous section, I wasn't able to observe a significant difference between using different block sizes. I based my comparison on block size of 128 vs. 32, yet I observed fps values that are almost the same for both block sizes.


