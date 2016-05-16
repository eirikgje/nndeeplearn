#ifndef NETWORK_H
#define NETWORK_H
#include <vector>
#include <random>
#include <tuple>

typedef std::vector<double> activations_t;
typedef std::tuple<activations_t, activations_t> trainunit_t;
typedef std::vector<trainunit_t> traindata_t;
typedef std::tuple<activations_t, int> testunit_t;
typedef std::vector<testunit_t> testdata_t;

typedef std::vector<double> layernodes_t;
typedef std::vector<layernodes_t> networknodes_t;
typedef std::vector<std::vector<double>> layerconnections_t;
typedef std::vector<layerconnections_t> networkconnections_t;
typedef std::tuple<networknodes_t, networkconnections_t> network_t;

class Network {
    private:
        std::vector<std::size_t> sizes;
        std::size_t num_layers;
        networknodes_t biases;
        networkconnections_t weights;
        std::default_random_engine generator;
        void update_mini_batch(
                const traindata_t::const_iterator& batch_start,
                const traindata_t::const_iterator& batch_end,
                const double& eta);
        network_t backprop(
                const activations_t& x,
                const activations_t& y);
        double cost_derivative(
                const double& output_activation,
                const double& y);
        int evaluate(const testdata_t& test_data);
        activations_t feed_forward(const activations_t& a);
    public:
        Network(const std::vector<std::size_t>& sizes, const unsigned& seed);
        void SGD(
                const traindata_t& training_data,
                const int& num_epochs,
                const size_t& mini_batch_size,
                const double& eta,
                const testdata_t& test_data={});
        void output_biases();
        void output_weights();
};
#endif
