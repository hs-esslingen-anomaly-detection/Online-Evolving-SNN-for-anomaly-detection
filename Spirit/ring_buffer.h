//
// Created by tobias on 21.02.20.
//

#ifndef SPIRIT_RING_BUFFER_H
#define SPIRIT_RING_BUFFER_H


#include <vector>
#include <memory>

template <class T> class ring_buffer{
public:
    explicit ring_buffer<T>(unsigned int max_size) {
        this->max_size = max_size;
        this->buffer = std::unique_ptr<T[]>(new T[max_size]);
        for(auto i = 0; i < max_size; i++) {
            buffer[i] = 0.0;
        }
    }

    explicit ring_buffer() = default;

    const std::vector<T>& get_values() const { return buffer;}

    double update(T new_value) {
        buffer[position] = new_value;
        position = (position+1)%max_size; //update position
        auto last_value = buffer[position];     //value at next position is first element in buffer

        sum = sum + new_value - last_value;

        if(num_items < max_size) {
            num_items++;
        }

        return sum/num_items;
    }

private:
    std::unique_ptr<T[]> buffer;
    unsigned int max_size {};
    unsigned int position {};
    double num_items {};
    double sum {};
};

#endif //SPIRIT_RING_BUFFER_H
