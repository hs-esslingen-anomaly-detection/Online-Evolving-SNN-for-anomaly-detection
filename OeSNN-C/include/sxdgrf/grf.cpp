//
// Created by tobias on 27.03.20.
//
#include <iostream>
#include "grf.h"

grf::grf(int input_neurons, int dim, double alpha,int avg_window, bool debug) :
    kmc(input_neurons, alpha, debug), m_avg(input_neurons, moving_average(avg_window, dim)) {
    this->n_input = input_neurons;
    this->n_dim = dim;
    this->alpha = alpha;
    this->debug = debug;
    this->marginal_distributions = std::vector<std::normal_distribution<double>>(
            dim,
            std::normal_distribution<double>(0, 0));

    init_covariances(n_input, n_dim);
}

std::vector<double> grf::update(const std::vector<double> &x){
    return update(Eigen::VectorXd::Map(x.data(), x.size()));
}

std::vector<double> grf::update(const Eigen::VectorXd& x) {
    kmc.update(x);
    auto idx = kmc.get_last_updated_idx();
    update_covariances(x, idx);
    update_marginal_distributions(idx);
    return get_excitement_values(x);
}

void grf::init_covariances(unsigned int size, unsigned int dim) {
    covariances = std::vector<Eigen::MatrixXd>(size);
    for(auto & covariance : covariances) {
        covariance = Eigen::MatrixXd::Zero(dim, dim);
    }
}

void grf::update_covariances(const Eigen::VectorXd& x, unsigned int idx) {
    auto alpha_val = (alpha == STATIONARY) ? 1.0/n : alpha;
    auto clusters = kmc.get_clusters();

    covariances[idx] = (1-alpha_val)*(covariances[idx]+alpha_val
                *(x-clusters[idx].get_center())
                *(x-clusters[idx].get_center()).transpose());

    n++;

    //for(unsigned int i = 0; i < clusters.size(); i++) {
        //covariances[i] = m_avg[i].update(covariances[i]);}
}

void grf::update_marginal_distributions(unsigned int cluster_idx) {
    auto cluster = kmc.get_clusters()[cluster_idx];
    const auto& mu = cluster.get_center();
    auto cov = covariances[cluster_idx];

    for (unsigned int i = 0; i < marginal_distributions.size(); i++) {
        marginal_distributions[i] = std::normal_distribution<double>(mu[i], cov(i, i));
    }
}

std::vector<double> grf::get_excitement_values(const Eigen::VectorXd &x) {
    auto exc_vals = std::vector<double>(n_input,  0.0);

    auto clusters = kmc.get_clusters();
    for(unsigned int i = 0; i < covariances.size(); i++) {
        if(i < clusters.size()) {
            if(covariances[i].isZero()) {
                continue;
            }

            double exc = std::exp((-0.5
                    *(x-clusters[i].get_center()).transpose()
                    *covariances[i].inverse()
                    *(x-clusters[i].get_center())));

            exc_vals[i] = exc;
        }
    }

    return exc_vals;
}