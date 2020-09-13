**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Zijing Peng
  - [LinkedIn](https://www.linkedin.com/in/zijing-peng/)
  - [personal website](https://zijingpeng.github.io/)
* Tested on: Windows 22, i7-8750H@ 2.22GHz 16GB, NVIDIA GeForce GTX 1060

![](/images/demo.gif)

### Performance Analysis



![](/images/chart1.png)



- **For each implementation, how does changing the number of boids affect performance? Why do you think this is?**

  When the number of biods increase, it takes longer time to perform each execute. Based on the profiling, we can find that at first, the average time grows slowly because there are only a few neighborhoods of one boid. However, when there are more and more biods, the average time grows based on n^2. For native implement, it suffers huge performance loss when the biods becomes larger and larger, and even reaches the limit of our hardware. Uniform Grid is much better that the naive implement, but it start to grow faster when number of boids become kind of large. Coherent Grid is the best one, even the number of boids grow to 500k, it still has good performance.

  ![](/images/chart2.png)

- **For each implementation, how does changing the block count and block size affect performance? Why do you think this is?**

  I tried 32, 64 and 128 for block count, and the result shows when the block size increases, the performance loses. Based on the results, I guess the best performance should be less than 32 or between 32 to 64. The performance relative to the block count should be related to the hardware, while I am using an old laptop with limited hardware.

  ![](/images/chart3.png)

- **For the coherent uniform grid: did you experience any performance improvements with the more coherent uniform grid? Was this the outcome you expected? Why or why not?**

  Yes. I test with grid width double and third times, which means the 1/8 and 1/27 grids.When grids number decrease, the performance lose. It might because when there are less grids, the volume of per grid increase, and it contains more boids in a grid. So it will take longer time to check if a boid in the grids is within the neighborhood distance. Although it will take more time shuffling and sorting when there are more grids, but seems that it is trivial compared to neighborhood distance checking.

  ![](/images/chart4.png)

- **Did changing cell width and checking 27 vs 8 neighboring cells affect performance? Why or why not? Be careful: it is insufficient (and possibly incorrect) to say that 27-cell is slower simply because there are more cells to check!**

  Yes. Based on our profiling, the performance slightly improves when changing to 27 cells. When we change the neighborhood counts, we also change the width of grids and the total grids numbers. Consider a volume with with of minimum distance, and we can regard it as a unit volume. When checking 8 neighbors, we will check 64 unit volume of boids. However, when checking 27 neighborhoods, the width of each neighbor halved, so we only check 27 unit volume of boids. Although we need to check more cells when changing to 27, but it is relatively trial.