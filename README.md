## CUDA Flocking Simulation

<img src="images/coherent_50000.gif" width = 600>

*Boids simulation running in Coherent Uniform Grid mode - 50000 particles (FPS capped to 35)*

* **Name:** Gizem Dal
  * [LinkedIn](https://www.linkedin.com/in/gizemdal), [personal website](https://www.gizemdal.com/)
  
### Project Summary

This is an optimized flocking simulation based on the Reynolds Boids algorithm with uniform grid with semi-coherent memory access. The simulation includes toggleable naïve and simple uniform grid modes for performance analysis purposes.
  
### Implementation Thoughts

Increasing the number of boids slows down the performance in all modes (naive, uniform grid, coherent uniform grid) while decreasing it improves performance. In every simulation step, we find the neighbors of every single boid and update velocities and positions. In naive mode, the complexity scales up pretty quickly since our naive method iterates over all the boids to find neighbors for every single boid. We can observe a significant difference between result of increasing boid count in naive mode versus coherent/uniform grid modes since the code is significantly optimized to only iterate over "likely neighboring" boids within a grid range.

I experienced about a 20-30 fps improvement on the performance with the coherent uniform grid. I believe the reason for that is because we're reshuffling the position and velocity buffers to have boid data sorted and this allows the device to access this sequential data more efficiently versus accessing data scattered in memory.

Decreasing the cell width (thus checking for more than 8 neighboring cells) slows down the performance. Besides having to check more cells, we also have to check for more neighbor boids when we decrease our cellWidth. Decreasing the cell width while keeping the search radius constant results in having to account for more neighboring cells, thus we increase the possible number of boids that are within our search radius. If we also increase the number of boids (but keep the scene_scale the same), every cell has a higher chance of enclosing more boids compared to before. All of these conditions can impact the performance when we check 27 vs 8 neighboring cells.

### Performance Analysis

The performance of the simulation is tested on a **Predator G3-571 Intel(R) Core(TM) i7-7700HQ CPU @ 2.80 GHz 2.81 GHz** machine.

I set a CUDA event timer inside the step simulation functions for all 3 modes and recorded the elapsed time between before and after the kernels are run to compute the new velocities and update the positions of all the particles. Here are the results I have observed with *N = 10000, blockSize = 128, VISUALIZATION = 1*:

Mode | Naive | Naive Uniform | Semi-Coherent Uniform
:---: | :---: | :---: | :---:
Time (in ms) | 15-16 | 2.7-3.4 | 2.6-3.2
