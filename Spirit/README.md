# spirit-lib-cpp

[![Generic badge](https://img.shields.io/badge/Made_with-C++-GREEN.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/Stable-No_release-BLUE.svg)](https://shields.io/)

## Description

sprit-lib-cpp is an C++ implementation with an additional python interface  of the Streaming Pattern Discovery in MultipleTime-Series (SPIRIT) algorithm as proposed by Papadimitriou et. al. [1] with a modified outlier scoring system adapted by Ahmad et. al. [2]. The library was implemented using the high level C++ template library Eigen for efficient linear algebra operations [3]. 


## Getting Started

```properties
git clone git@github.com:tobiaskortus/spirit-lib-cpp.git
```

```bat
mkdir build
cd build
cmake ..
make
```

## API

### Constructor

```cpp
spirit(int dimensionality, int window_size, int initial_eigencomponents,
       double exp_forgetting_factor, double lower_energy_threshold, 
       double upper_energy_threshold, bool debug);
```

|Parameter|Type|Description|
|---|---|---|
|`dimensionality`|`int`|dimensionality of the input data (d>=2)|
|`window_size`|`int`|window size used for the outlier score (<img src="https://render.githubusercontent.com/render/math?math=\tilde{\mu_F}">) [2]|
|`initial_eigencomponents`|`int`|initial number of initialized eigencomponents|
|`exp_forgetting_factor`|`double`|exponential forgetting factor <img src="https://render.githubusercontent.com/render/math?math=0 < \lambda < 1"> used for following trend drifts over time (default = 0.96) as defined in Papadimitriou et. al. [1]|
|`lower_energy_threshold`|`double`|lower threshold used for adaption of required eigencomponents (default = 0.95) as defined in Papadimitriou et. al. [1]|
|`upper_energy_threshold`|`double`|upper threshold used for adaption of required eigencomponents (default = 0.98) as defined in Papadimitriou et. al. [1]|
|`debug`|`bool`|enables additional debugging features such as the calculation of the reconstruction performed by the SPIRIT algorithm|
</br>

### Processing a whole Series

```cpp
std::vector<double> spirit::run(const std::vector<Eigen::VectorXd> &series);
std::vector<double> spirit::run(const std::vector<std::vector<double>> &series);
```

|Parameter|Type|Description|
|---|---|---|
|`series`|`const std::vector<Eigen::VectorXd>&`|Given timeseries either as a vector of Eigen::Vectors or nested vector of doubles|
|`series`|`const std::vector<std::vector<double>>&`|Given timeseries either as a vector of Eigen::Vectors or nested vector of doubles|

|Returns|Type|Description|
|---|---|---|
||`std::vector<double>`|Calculated anomaly score for the given timeseries|
</br>

### Processing a single Datapoint

```cpp
double spirit::update(const Eigen::VectorXd& x);
double spirit::update(const std::vector<double>& x);
```

|Parameter|Type|Description|
|---|---|---|
|`x`|`const Eigen::VectorXd&`|Given datapoint either as a  Eigen::Vectors or a vector of doubles|
|`x`|`const std::vector<double>&`|Given datapoint either as a  Eigen::Vectors or a vector of doubles|

|Returns|Type|Description|
|---|---|---|
||`double`|Calculated anomaly score for the given datapoint|
</br>

### Debugging Functionality

```cpp
[[nodiscard]] std::vector<Eigen::VectorXd> get_reconstruction();
```

|Returns|Type|Description|
|---|---|---|
||`std::vector<Eigen::VectorXd>`|Predicted reconstruction for the given timeseries/ datapoints up to the current timestep|

## References

[1] Spiros Papadimitriou, Jimeng Sun, and Christos Faloutsos. 2005. Streaming pattern discovery in multiple time-series. In Proceedings of the 31st international conference on Very large data bases (VLDB ’05). VLDB Endowment, 697–708. (http://www.cs.cmu.edu/~jimeng/papers/spirit_vldb05.pdf)

[2] Ahmad, Subutai and Scott Purdy. “Real-Time Anomaly Detection for Streaming Analytics.” ArXiv abs/1607.02480 (2016): n. pag. (https://arxiv.org/pdf/1607.02480.pdf)

[3]  http://eigen.tuxfamily.org/, (https://gitlab.com/libeigen/eigen)

[4] https://pybind11.readthedocs.io/en/stable/intro.html, (https://github.com/pybind/pybind11)
