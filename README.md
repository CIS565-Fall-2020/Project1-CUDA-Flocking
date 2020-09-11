**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Xuecheng Sun
  * https://www.linkedin.com/in/hehehaha12138/
* Tested on: Windows 10, R7-3700X @ 3.6GHz 32GB, RTX 2070 Super 8GB

### Result

Naive Flocking Gif

![](./images/Naive%20Visualized.gif)

Uniform Grid Gif

![](./images/Uniform%20Grid%20Visualized.gif)

Coherent Grid Gif

![](./images/Coherent%20Grid%20Visualized.gif)



### Performance Analysis

#### Elapsed Time After a Single Step

![](./images/Elapsed%20Time%20Chart.png)



#### Frame Rates Data for Different  Methods when Simulating Different Numbers of Boids

![](./images/Frame%20Rate%20Chart.png)

**Answer for Q1:**This chart shows fps data with different boids counts and different simulation methods. When the program simulate less boids, the speed differences is not so significant. However, when we simulate a really huge number of boids, the efficiency gap is huge. For naive method, it only can run 0.5 fps when simulating boids, but for Coherent it still have 230 frames per second.

This could because the relative differences of loop times is bigger when we have more boids to simulate. Considering we have 500000 boids need to be simulate, for naive simulation, the program will loop all 500000 boids, but for coherent method, it could only loop 500 times in one step. If we have 5000 boids to simulate the maximum differences is 5000, in comparison, in 500000 boids simulation, the difference could be a hundred times more than the former.

#### Frame Rates Data for Different Methods under Different Block Size

![](./images/FPS%20Block%20Size.png)

**Answer for Q2:** Next I try to run this program under different block size settings. The result is really interesting because it is not monotonous. The performance are almost the same when we are using block size 32 to 256. But for lower block size (16), the program have a significant performance loss.  This could because the default warp size is 32 thread. If we call the function under the situation of 16 threads per block, this will let the system cannot fully use the whole warp as a computation unit. For higher count it is really tricky,  it also suffer some performance loss, this could because the SM can only deal with a warp (32 threads) at the same time, bigger block size means more warp need to be used for a block. Among all the block size setting, the 128 block size have the best performance, perhaps it find a balance between warp size and block size.

#### Different Coherent Grid Size

![](./images/Grid%20Count.png)

**Answer for Q3:** I shrink the grid cell length to double the grid quantities in the simulation. This shows that program suffered performance loss when we have more grid to deal with. This could because the sort function and reshuffle we used will take more time to deal with longer data. 



#### Different Searching Radius

![](./images/FPS%20Block%20Search.png)

**Answer for Q4:** For coherent grid, this will not cause significant performance loss. In comparison, the program suffers performance loss when we have a bigger search radius. This could because the coherency of the data structure in uniform grid is bad. More search radius means worse coherent data access for position and velocity data. For coherent grid, because of coherent construction of data, this cannot cause a big problem of performance.