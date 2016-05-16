#ifndef MNIST_READ_H
#define MNIST_READ_H

#include <vector>
#include <string>

int reverse_int(const uint32_t& i);

std::vector<std::vector<double>> my_read_mnist_imfile(
        const int& num_to_read,
        const std::string& filepath,
        const int& start_index);

std::vector<int> my_read_mnist_labelfile(
        const int& num_to_read,
        const std::string& filepath,
        const int& start_index);


std::vector<std::tuple<std::vector<double>, std::vector<double>>>
massage_training_data(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels);

std::vector<std::tuple<std::vector<double>, int>>
massage_test_data(
        const std::vector<std::vector<double>>& images,
        const std::vector<int>& labels);
//void read_mnist(const int& num_images, const int& num_elements,
//        std::vector<std::vector<double>>& arr,
//        const std::string& filepath);

#endif
