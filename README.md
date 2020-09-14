**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**

* Jacky Lu
  * [LinkedIn](https://www.linkedin.com/in/jacky-lu-506968129/)

# README

## Result:
* ### Naive Implementation (5,000 Boids & 10,648 Cells)
![](images/naive_5000_boids_10648_cells.gif)
* ### Uniform Grid Implementation With Scattered Data [27 Neighboring Cells Search] (5,000 Boids & 10,648 Cells)
![](images/uniform_27_5000_boids_10648_cells.gif)
* ### Uniform Grid Implementation With Scattered Data [8 Neighboring Cells Search] (5,000 Boids & 10,648 Cells)
![](images/uniform_8_5000_boids_10648_cells.gif)
* ### Uniform Grid Implementation With Coherent Data [27 Neighboring Cells Search] (5,000 Boids & 10,648 Cells)
![](images/coherent_27_5000_boids_10648_cells.gif)
* ### Uniform Grid Implementation With Coherent Data [8 Neighboring Cells Search] (5,000 Boids & 10,648 Cells)
![](images/coherent_8_5000_boids_10648_cells.gif)
* ### [Uniform Grid Implementation With Coherent Data [8 Neighboring Cells Search] (20,000,000 Boids & 64,964,808 Cells)](https://drive.google.com/file/d/100R0v3XGLHOtpD7fnWTzJrbg_4Zd0yig/view?usp=sharing)

* #### P.S. I wasn't able to perform much performance analysis due to the lack of time because I have been trying to find the bug that crashes my program for the entire weekend. However, as a result of the lengthy debugging process, I was able to have a lot of practice and hands-on experience with the Nsight debugger. I find the Nsight debugger very helpful. By tracking global buffer's data in Warp Watch, I was able to locate the unusual data pattern in dev_particleArrayIndices, and ultimately found out that I freed the buffers' device memories at the wrong place in the program.