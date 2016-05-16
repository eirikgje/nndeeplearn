#include "network.h"
#include "mnist_read.h"
#include <chrono>
#include <iostream>
#include <random>

int main() {

    auto train_image_vec = my_read_mnist_imfile(50000, "/home/eirik/data/machinelearning/neuralnets/train-images-idx3-ubyte", 0);
    auto train_label_vec = my_read_mnist_labelfile(50000, "/home/eirik/data/machinelearning/neuralnets/train-labels-idx1-ubyte", 0);
    auto test_image_vec = my_read_mnist_imfile(10000, "/home/eirik/data/machinelearning/neuralnets/t10k-images-idx3-ubyte", 0);
    auto test_label_vec = my_read_mnist_labelfile(10000, "/home/eirik/data/machinelearning/neuralnets/t10k-labels-idx1-ubyte", 0);
    auto training_data = massage_training_data(train_image_vec, train_label_vec);
    auto test_data = massage_test_data(test_image_vec, test_label_vec);
//    double hey = 0.23;
//    for (int i=0; i<image_vec[0].size(); i++){
//        std::cout << image_vec[0][i] << '\n';
//    }
//    std::cout << hey << '\n';
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::vector<std::size_t> sizes {784, 30, 10};
    Network my_network(sizes, seed);
    my_network.SGD(training_data, 30, 10, 3.0, test_data);
//    std::vector<double> input {0.4, 0.9, 0.1};
//    std::vector<double> res = my_network.feed_forward(input);
//    for (int i=0; i<res.size(); i++) {
//        std::cout << res[i] << '\n';
//    }
//    my_network.output_biases();
//    my_network.output_weights();
    return 0;
}
