//
// Created by tobias on 27.03.20.
//

#ifndef SXDGRF_LIB_CPP_GRF_H
#define SXDGRF_LIB_CPP_GRF_H

#define STATIONARY -1

#include <stdexcept>
#undef eigen_assert
#define eigen_assert(x) \
  if (!(x)) { throw (std::runtime_error("eigen_assert")); }

#include "kmeans/kmeans.h"
#include "utils/moving_average.h"
#include <Eigen/Dense>
#include <vector>
#include <random>

class grf {
public:
    explicit grf(int input_neurons, int dim, double alpha = STATIONARY, int avg_window = 1, bool debug = false);
    ~grf() = default;

    std::vector<double> update(const std::vector<double>& x);
    std::vector<double> update(const Eigen::VectorXd& x);

    //Only used for debug purposes
    [[nodiscard]] std::vector<kmeans::cluster> get_clusters() const { return kmc.get_clusters(); };
    [[nodiscard]] std::vector<Eigen::MatrixXd> get_covariance_matrices() const { return covariances; };
    [[nodiscard]] std::vector<std::normal_distribution<double>> get_marginal_distributions() const { return marginal_distributions; }

private:
    void init_covariances(unsigned int size, unsigned int dim);
    void update_covariances(const Eigen::VectorXd& x, unsigned int idx);
    void update_marginal_distributions(unsigned int cluster_idx);

    std::vector<double> get_excitement_values(const Eigen::VectorXd& x);

    double alpha;
    bool debug;
    unsigned int n_input;
    unsigned int n_dim;
    unsigned int n = 1;

    kmeans kmc;
    std::vector<std::normal_distribution<double>> marginal_distributions;
    std::vector<Eigen::MatrixXd> covariances;
    std::vector<moving_average> m_avg;
};


#endif //SXDGRF_LIB_CPP_GRF_H
