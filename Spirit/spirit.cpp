//
// Created by tobias on 21.02.20.
//

#include <iostream>
#include "spirit.h"

spirit::spirit(int dimensionality, int window_size, int initial_eigencomponents,
               double exp_forgetting_factor, double lower_energy_threshold, double upper_energy_threshold, 
               bool debug) {

    if (dimensionality < 2)
        throw std::invalid_argument("Invalid dimensionality (required: d>=2)");

    weights = Eigen::MatrixXd::Identity(dimensionality, dimensionality);
    d = Eigen::VectorXd::Zero(dimensionality);
    Ei = Eigen::VectorXd::Zero(dimensionality);

    moving_avg = ring_buffer<double>(window_size);

    this->exp_forgetting_factor = exp_forgetting_factor;
    this->lower_energy_threshold = lower_energy_threshold;
    this->upper_energy_threshold = upper_energy_threshold;
    this->aggregate = welford_aggregate();

    this->number_of_eigencomponents = initial_eigencomponents;
    this->dimensionality = dimensionality;
    this->debug = debug;

    // Initialized with fixed size for better performance
    this->x_prime = Eigen::MatrixXd::Zero(dimensionality, dimensionality);
    this->y = Eigen::VectorXd::Zero(dimensionality);
    this->e = Eigen::MatrixXd::Zero(dimensionality, dimensionality);
}

std::vector<double> spirit::run(const std::vector<Eigen::VectorXd> &series) {
    auto score = std::vector<double>(series.size());
    for(int i = 0; i < series.size(); i++) {
        score[i] = update(series[i]);
    }
    return score;
}

std::vector<double> spirit::run(const std::vector<std::vector<double>> &series) {
    auto score = std::vector<double>(series.size());
    for(int i = 0; i < series.size(); i++) {
        score[i] = update(series[i]);
    }
    return score;
}

double spirit::update(const std::vector<double>& x) {
    Eigen::VectorXd x_vec = Eigen::VectorXd::Map(x.data(), x.size());
    return update(x_vec);
}

double spirit::update(const Eigen::VectorXd& x) {
    this->aggregate.count+=1;

    x_prime.col(0)  = x;

    for (int i = 0; i < number_of_eigencomponents; i++) {

        y[i] = weights.col(i).transpose() * x_prime.col(i);
        d[i] = exp_forgetting_factor * d[i] + std::pow(y[i], 2);
        e.col(i) = x_prime.col(i) - y[i] * weights.col(i);

        //Update weights using gradient descent
        weights.col(i) = weights.col(i) + (1/d[i] * y [i]) * e.col(i);

        if (i+1 < number_of_eigencomponents) {
            x_prime.col(i+1) = e.col(i);
        }
    }

    if(debug) {
        reconstruction.push_back(x - e.col(number_of_eigencomponents - 1));
    }

    // Update energy values
    E = (exp_forgetting_factor * (aggregate.count-1)*E + x.squaredNorm())/aggregate.count;
    Ei.head(number_of_eigencomponents) = (Ei.head(number_of_eigencomponents) * exp_forgetting_factor * (aggregate.count-1)
                                        + y.head(number_of_eigencomponents).cwiseProduct(y.head(number_of_eigencomponents)))
                                        /aggregate.count;
    auto Ek = Ei.sum();

    //Adapt number of eigencomponents based on calculated energy values & thresholds
    if (Ek < lower_energy_threshold * E && number_of_eigencomponents < dimensionality) {
        number_of_eigencomponents += 1;
        //Set values of d[number_of_eigencomponents] as canonical unit vector e_{number_of_eigencomponents}
        weights.col(number_of_eigencomponents-1) *= 0;
        weights(number_of_eigencomponents-1, number_of_eigencomponents-1) = 1.0;
        //Reset corresponding values of Ei, di[number_of_eigencomponents] to zero
        Ei[number_of_eigencomponents-1] = 0;
        d[number_of_eigencomponents-1] = 0;
    } else if (Ek > upper_energy_threshold * E && number_of_eigencomponents > 1) {
        number_of_eigencomponents -= 1;
    }

    return calculate_score(e.col(number_of_eigencomponents-1).maxCoeff());
}

double spirit::calculate_score(const double& error) {
    update_welford(aggregate, error);
    finalize_welford(aggregate);
    auto standard_deviation = std::sqrt(aggregate.variance);
    auto windowed_mean = moving_avg.update(error);
    auto cdf = std::erfc(-(windowed_mean - aggregate.mean)/standard_deviation/std::sqrt(2))/2;
    return 2.0 * std::abs(cdf - 0.5);
}

void spirit::update_welford(spirit::welford_aggregate& aggregate, double x) {
    auto delta = (x - aggregate.mean);
    aggregate.mean += delta/aggregate.count;
    auto delta2 = (x - aggregate.mean);
    aggregate.m2 += delta * delta2;
}

void spirit::finalize_welford(spirit::welford_aggregate &aggregate) {
    //Avoid zero division and senseless results from division by -1
    if (aggregate.count < 2) {
        aggregate.variance = 0;
        aggregate.sample_variance = 0;
    }

    aggregate.variance = aggregate.m2/aggregate.count;
    aggregate.sample_variance = aggregate.m2/(aggregate.count-1);
}

spirit::~spirit() = default;