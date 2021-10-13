# sxdgrf-lib: multidimensional gaussian receptive field for streaming data

## Getting Started

#### Build

```bash
cd <source_dir>
mkdir build
cd build
cmake ..
make
```

## Documentation

#### Constructor

```C++
explicit grf(int input_neurons, int dim, double alpha = STATIONARY, bool debug = false)
```

- **input_neurons:** number of input neurons used for the following SNN. The size of the input neurons is equal to the size of created receptive fields.

- **dim**: dimensionality of the input data

- **alpha**: weighting factor of previous and current observation. Used for recursive algorithms updating the center of each clusters and the incremental estimate of the covariance matrixes used for the calculation of excitement values for each receptive field.

  - **stationary condition <img src="https://render.githubusercontent.com/render/math?math=(\alpha =\frac{1}{n})">**: all observations are weighted equally
  - **quasi-stationary condition <img src="https://render.githubusercontent.com/render/math?math=(\alpha = const.)">** newer observationa are weighted more than observations in the past. The observation window is defined approximately by <img src="https://render.githubusercontent.com/render/math?math=1/\alpha">. [2]

    </br>

    ![img](doc/averaging.png)

- **debug:** enable debug mode for grf and all subcomponents which enables the following additional functionality:

  - save and access elements of each kmeans cluster: `get_elements()`

  > Note: For additional information visit [online k-means](kmeans/README.md) documentation


    </br>

> Note: The clusters used for the receptive fields for each dimension are created incrementally based on incoming data points. For all timesteps t < input_neurons the excitement values for all neurons with an index smaller than the input neuron size are initialized to zero !

</br>

#### Update

```C++
std::vector<std::vector<double>> update(const std::vector<double>& x);
```

```C++
std::vector<std::vector<double>> update(const Eigen::VectorXd& x);
```

- **x**: input vector either as `const std::vector<double>&` or `const Eigen::VectorXd&`
- **returns**: excitement values for all dimensions

#### Getters

```C++
[[nodiscard]] std::vector<kmeans::cluster> get_clusters() const
```

- **returns:** all cluster elements produced by online kmeans algortithm (`kmeans::cluster`)
  > Note: For additional information visit [online k-means](kmeans/README.md) documentation

```C++
[[nodiscard]] std::vector<Eigen::MatrixXd> get_covariance_matrices() const
```

- **returns:** all covariance matrices for each receptive field as `std::vector<Eigen::MatrixXd>`

</br>

## Additional Material

- [sxdgrf-lib python module](python_module/README.md)
- [online k-means](kmeans/README.md)

## References

[1] Panuku L.N., Sekhar C.C. (2008) Region-Based Encoding Method Using Multi-dimensional Gaussians for Networks of Spiking Neurons. In: Ishikawa M., Doya K., Miyamoto H., Yamakawa T. (eds) Neural Information Processing. ICONIP 2007. Lecture Notes in Computer Science, vol 4984. Springer, Berlin, Heidelberg

[2] https://lmb.informatik.uni-freiburg.de/lectures/old_lmb/mustererkennung/WS0506/07_c_ME.pdf
