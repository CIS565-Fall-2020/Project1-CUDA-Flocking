**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Shenyue Chen
  * [LinkedIn](https://www.linkedin.com/in/shenyue-chen-5b2728119/), [personal website](http://github.com/EvsChen)
* Tested on: Windows 10, Intel Xeon Platinum 8259CL @ 2.50GHz 16GB, Tesla T4 (AWS g4dn-xlarge)


<p align="center">
<image src="docs/capture.png" />
<image src="docs/animation.gif" />
</p>

### Performance analysis
In the performance analysis section, I measured the FPS under the release mode to get the performance for the three implementations.

1. For all the implementations, increasing the number of boids will decrease the performance, since increasing the number of boids will significantly increase the computation loads.

<p align="center">
<image src="docs/boid-count.png" />
</p>

2. For all the implementations, changing the block size does not impact the performance significantly.

<p align="center">
<image src="docs/block-size.png" />
</p>

3. For small boids numbers (i.e. 5000), the coherent grid does not perform much better than the uniform grid. However, when the number of boids gets bigger (i.e. 500000), the coherent grid is clearly much faster than the uniform grid. 

The reason for this is that we access the memory for the coherent grid in a a almost sequential manner, while in the uniform grid, the memory access is random.

4. Changing the cell width will affect the performance. When we decrease the cell width, the number of cells we need to check increase from 8 to 27. However, the number of boids contained in this total volume decreases. Assuming the boids are distributed uniformly,

- for 1 unit width, V = 27 * 1^3 = 27,
- for 2 unit width, V = 8 * 2 ^ 3 = 64.

So the performance will be better for the 27-cell checking.


5. Turn off the visualization will increase the performance for both uniform and coherent grid, but not for the naive method. The reason for this is that the visualization is not a bottleneck for the naive method.
<p align="center">
<image src="docs/visualization.png" />
</p>


