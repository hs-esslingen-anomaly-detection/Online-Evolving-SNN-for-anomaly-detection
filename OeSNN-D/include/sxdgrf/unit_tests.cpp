//
// Created by tobias on 24.03.20.
//
#define  CATCH_CONFIG_MAIN

#include <catch.hpp>
#include <Eigen/Dense>
#include "kmeans/kmeans.h"
#include "utils/moving_average.h"

TEST_CASE("Average is calculated correctly", "[movavg]") {
    std::vector<Eigen::MatrixXd> matrices;

    for (auto i = 0; i < 101; i++) {
        auto m = Eigen::MatrixXd(2, 2);
        m << i, i, i, i;
        matrices.push_back(m);
    }

    auto avg = Eigen::MatrixXd(2, 2);
    avg << 50, 50, 50, 50;

    auto mo_avg = moving_average(2, 2);

    SECTION("Average is calculated correct for first element") {
        REQUIRE(mo_avg.update(matrices[0]) == matrices[0]);
    }

    SECTION("Average is calculated correct for second element") {
        for(auto i = 0; i < matrices.size()-1; i++) {
            mo_avg.update(matrices[i]);
        }

        REQUIRE(mo_avg.update(matrices[matrices.size()-1]) == avg);
    }
}

TEST_CASE("Kmeans cluster is created correctly", "[cluster]") {
    auto p1 = Eigen::VectorXd(3);
    auto p2 = Eigen::VectorXd(3);
    auto new_center = Eigen::VectorXd(3);

    p1 << 1, 1, 1;
    p2 << 2, 2, 2;
    new_center << 1.5, 1.5, 1.5;

    auto c = kmeans::cluster(p1, STATIONARY);

    SECTION("Cluster is created with correct center and n == 1") {
        REQUIRE(c.get_center() == p1);
        REQUIRE(c.get_number_of_items() == 1);
    }

    SECTION("Cluster is updated correctly") {
        c.update(p2);
        REQUIRE(c.get_center() == new_center);
        REQUIRE(c.get_number_of_items() == 2);
    }
}
