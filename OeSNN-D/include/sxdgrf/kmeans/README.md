# sxdgrf-lib: online k-means

TODO: Introduction 

```c
1. initialize the k cluster centers z1, ..., zk in any way
2. create counters n1, ..., nk and initialize them to zero
3. loop
4.    get new data point x
5.    determine the closest center zi to x
6.    update the number of points in that cluster
7.    update the cluster center
8. end loop
```
**Listing 1:** Pseudocode for Online Lloyd’s Algorithm [1]

### Update of the cluster center
The center of a cluster is updated using a recursive approach in order to sustain the online ability of the algorithm. For each additional point in a cluster the number of points <img src="https://render.githubusercontent.com/render/math?math=n"> is increased by one and the center is updated using the following equation: <img src="https://render.githubusercontent.com/render/math?math=\mu_{n}=\mu_{n-1} + \alpha(x_{n}-\mu_{n-1}) = (1-\alpha)\mu_{n-1}+\alpha x_{n}"> where <img src="https://render.githubusercontent.com/render/math?math=\alpha"> represents a weighting factor of the previous and current observation. A distinction is made between the following two different conditions:

- **stationary condition <img src="https://render.githubusercontent.com/render/math?math=(\alpha =\frac{1}{n})">**: all observations are weighted equally
- **quasi-stationary condition <img src="https://render.githubusercontent.com/render/math?math=(\alpha = const.)">** newer observations are weighted more than observations in the past. The observation window is defined approximately by <img src="https://render.githubusercontent.com/render/math?math=1/\alpha">. [2]

</br>
    
![img](../doc/averaging.png)






[1] Morissette, Laurence & Chartier, Sylvain. (2013). The k-means clustering technique: General considerations and implementation in Mathematica. Tutorials in Quantitative Methods for Psychology. 9. 15-24. 10.20982/tqmp.09.1.p015. 

[2] https://lmb.informatik.uni-freiburg.de/lectures/old_lmb/mustererkennung/WS0506/07_c_ME.pdf
