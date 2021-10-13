//
// Created by tobias on 22.03.20.
//

#include "kmeans.h"

kmeans::kmeans(int k, double alpha, bool debug) : k(k), alpha(alpha), debug(debug) {}

std::vector<kmeans::cluster> kmeans::update(const std::vector<double>& x) {
    return update(Eigen::VectorXd::Map(x.data(), x.size()));
}

std::vector<kmeans::cluster> kmeans::update(const Eigen::VectorXd& x) {
    if(clusters.size() < k) {
        clusters.emplace_back(cluster(x, alpha, debug, debug_idx));
        return clusters;
    }

    auto idx = combine_clusters(x, clusters);
    clusters[idx].update(x, debug_idx);
    last_updated_idx = idx;
    debug_idx++; //Only used for debug purposes
    return  clusters;
}

int kmeans::combine_clusters(const Eigen::VectorXd &x, const std::vector<cluster> &cl) {
    auto aggregate = calculate_min_dist(x, cl);
    auto cluster = aggregate.first;
    return cluster;
}

std::pair<int, double> kmeans::calculate_min_dist(const Eigen::VectorXd& x, const std::vector<cluster>& cl) {
    auto min_dist = std::numeric_limits<double>::max();
    auto min_cluster_id = -1;

    for(unsigned int i = 0; i < cl.size(); i++) {
        auto dist = euclidean(x, cl[i].get_center());
        if(min_dist > dist) {
            min_dist = dist;
            min_cluster_id = i;
        }
    }

    return std::pair<int, double>(min_cluster_id, min_dist);
}

double kmeans::euclidean(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
    return (x-y).rowwise().squaredNorm()[0];
}
