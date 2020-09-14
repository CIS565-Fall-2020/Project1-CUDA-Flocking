<h1> CUDA Flocking Simulation with Reynold Boids Algorithm

# ![top](images/top_image.png)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 1 - Flocking**
* Haorong Yang
* [LinkedIn](https://www.linkedin.com/in/haorong-henry-yang/)
* Tested on: Windows 10 Home, i7-10750H @ 2.60GHz 16GB, GTX 2070 Super Max-Q (Personal)



This is a CUDA based simulation of the Reynold Boids Algorithm. Three approaches were taken to perform neighbor search.
The Naive approach does a brute force search through every other boid to update one boid.
The Uniform Grid and Coherent Grid searches through the cells within its search radius, 
but Coherent Grid has a modification in the method of accessing elements.

The method used to compare performance was by frame rate under different boid counts.

Comparison of 3 methods of step simulation (Naive Search, Uniform Grid, Coherent Grid):
![chart1](images/fpsGraph8.PNG)

