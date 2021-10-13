//
// Created by tobias on 21.02.20.
//

#ifndef OeSSN_MOVING_AVERAGE_H
#define OeSSN_MOVING_AVERAGE_H


#include <vector>
#include <memory>

template <class T> class MovingAverage{
public:
    explicit MovingAverage<T>(unsigned int max_size) {
        this->position = 0;
        this->num_items = 0;
        this->sum = 0;
        this->max_size = max_size;
        this->buffer = new T[max_size];

        for(auto i = 0; i < max_size; i++) {
            buffer[i] = 0.0;
        }
    }

    explicit MovingAverage() = default;

    ~MovingAverage() {
        delete buffer;
    }

    const std::vector<T>& getValues() const { return buffer;}

    double UpdateAverage(T new_value) {
        buffer[position] = new_value;
        position = (position+1)%max_size;       //update position
        auto last_value = buffer[position];     //value at next position is first element in buffer

        sum += new_value - last_value;

        if(num_items < max_size) {
            num_items++;
        }

        return sum/num_items;
    }

private:
    T* buffer;
    unsigned int max_size {};
    unsigned int position {};
    double num_items {};
    double sum {};
};

#endif //OeSSN_MOVING_AVERAGE_H
