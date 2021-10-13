/**
 * A simple test application which can be used for debugging
 */

#include <iostream>
#include <fstream>
#include <chrono>
#include <Eigen/Dense>

#include "spirit.h"


std::vector<Eigen::VectorXd> load_default_data() {
    std::vector<Eigen::VectorXd> data;

    for(auto i = 0; i < 100; i++) {
        Eigen::VectorXd timestep(3);
        timestep << ((double) rand() / (RAND_MAX)), 
                    ((double) rand() / (RAND_MAX)), 
                    ((double) rand() / (RAND_MAX));
        data.push_back(timestep);
    }


    return data;
}

int main() {
    std::vector<Eigen::VectorXd>data = load_default_data();

    spirit spirit(3, 100);

    for (int i = 0; i < 1; i++) {
        for (const Eigen::VectorXd& timestep : data) {
            std::cout << spirit.update(timestep) << std::endl;
        }
    }
    return 0;
}