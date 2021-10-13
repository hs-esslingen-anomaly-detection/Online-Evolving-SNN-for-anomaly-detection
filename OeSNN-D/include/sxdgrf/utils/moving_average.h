//
// Created by tobias on 21.02.20.
//

#ifndef SXDGRF_LIB_CPP_MOVING_AVERAGE_H
#define SXDGRF_LIB_CPP_MOVING_AVERAGE_H


#include <vector>
#include <memory>
#include <iostream>

class moving_average{
public:
    explicit moving_average(unsigned int max_size, unsigned int dim) {
        this->max_size = max_size;
        this->sum = Eigen::MatrixXd::Zero(dim, dim);
        this->buffer = std::vector<Eigen::MatrixXd>(max_size ,Eigen::MatrixXd::Zero(dim, dim));
    }

    explicit moving_average() = default;

    Eigen::MatrixXd update(const Eigen::MatrixXd& new_value) {
        buffer[position] = new_value;
        auto first_position = (position+1)%max_size; //update position
        auto first_element = buffer[first_position];     //value at next position is first element in buffer

        sum += new_value - first_element;

        if (num_items < max_size)
            num_items++;

        position = first_position;
        return sum/num_items;
    }

private:
    std::vector<Eigen::MatrixXd> buffer{};
    unsigned int max_size {};
    unsigned int position {};
    double num_items {};
    Eigen::MatrixXd sum {};
};

#endif //SXDGRF_LIB_CPP_MOVING_AVERAGE_H
