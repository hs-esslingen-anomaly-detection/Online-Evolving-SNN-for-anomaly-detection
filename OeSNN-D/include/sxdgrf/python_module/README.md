# sxdgrf-lib: python module

## Prerequisites:

## Build:

```sh
cd <source_dir>
mkdir build
cd build
cmake ..
make
```

## Usage:

```py
from utils.cluster_utils import distance
from utils.cov_utils import confidence_ellipse, multivariate_normal
from grf_module import grf, cluster
```


```py
g = grf(2, 2, -1)

for i, x in enumerate(data):
    g.update(x)

cov = g.get_covariance_matrices()[0]
pos = g.get_clusters[0].get_position()
x = pos[0]
y = pos[1]

confidence_ellipse(x, y, cov, ax, n_std=1, edgecolor='red')
multivariate_normal(pos, cov, x)
```