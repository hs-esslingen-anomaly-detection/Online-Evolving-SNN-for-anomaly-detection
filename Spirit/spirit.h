//
// Created by tobias on 21.02.20.
//

#ifndef SPIRIT_SPIRIT_H
#define SPIRIT_SPIRIT_H

#include <cmath>
#include <deque>
#include <numeric>
#include <Eigen/Dense>

#include "ring_buffer.h"

class spirit {
public:
    spirit(int dimensionality,
           int window_size,
           int initial_eigencomponents = 1,
           double exp_forgetting_factor = 0.96,
           double lower_energy_threshold = 0.95,
           double upper_energy_threshold = 0.98,
           bool debug = false);

    std::vector<double> run(const std::vector<Eigen::VectorXd>& series);
    std::vector<double> run(const std::vector<std::vector<double>>& series);

    double update(const std::vector<double>& x);
    double update(const Eigen::VectorXd& x);

    [[nodiscard]] std::vector<Eigen::VectorXd> get_reconstruction() { return reconstruction; };

    ~spirit();

private:
    struct welford_aggregate {
        unsigned int count;
        double mean{};
        double m2{};
        double variance{};
        double sample_variance{};
    };

    static inline void update_welford(welford_aggregate& aggregate, double x);
    static inline void finalize_welford(welford_aggregate& aggregate);
    double calculate_score(const double& error);

    double exp_forgetting_factor {};
    double lower_energy_threshold {};
    double upper_energy_threshold {};

    double E {};

    int dimensionality {};
    int number_of_eigencomponents {};

    bool debug {};

    welford_aggregate aggregate;

    Eigen::MatrixXd weights ;
    Eigen::VectorXd d;
    Eigen::VectorXd Ei;

    //Variables for update routine
    Eigen::MatrixXd x_prime;
    Eigen::VectorXd y;
    Eigen::MatrixXd e;

    ring_buffer<double> moving_avg;

    std::vector<Eigen::VectorXd> reconstruction{};
};

#endif //SPIRIT_SPIRIT_H
