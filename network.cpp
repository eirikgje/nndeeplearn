#include "network.h"
#include "calculations.h"
#include "utils.h"

#include <iostream>
#include <boost/format.hpp>

Network::Network(const std::vector<std::size_t>& sizes, const unsigned& seed) :
    sizes(sizes),
    num_layers(sizes.size()),
    generator(seed)
{
    std::normal_distribution<double> distribution(0.0, 1.0);
    layernodes_t currbias;
    layerconnections_t currto_nodes;
    std::vector<double> currfrom_nodes;
    for (int i=1; i<num_layers; i++) {
        for (int j=0; j<sizes[i]; j++) {
            currbias.push_back(distribution(generator));
            for (int k=0; k<sizes[i-1]; k++) {
                currfrom_nodes.push_back(distribution(generator));
            }
            currto_nodes.push_back(currfrom_nodes);
            currfrom_nodes.clear();
        }
        biases.push_back(currbias);
        currbias.clear();
        weights.push_back(currto_nodes);
        currto_nodes.clear();
    }

}


void Network::output_biases() {
    for (int i=0; i<biases.size(); i++) {
        for (int j=0; j<biases[i].size(); j++) {
            std::cout << biases[i][j] << '\n';
        }
    }
}


void Network::output_weights() {
    for (int i=0; i<weights.size(); i++) {
        for (int j=0; j<weights[i].size(); j++) {
            for (int k=0; k<weights[i][j].size(); k++) {
                std::cout << weights[i][j][k] << '\n';
            }
        }
    }
}


activations_t Network::feed_forward(const activations_t& a) {
    activations_t curr_from_values(a);
    activations_t curr_to_values;
    for (int i=1; i<num_layers; i++) {
        size_t num_from_nodes = sizes[i-1];
        size_t num_to_nodes = sizes[i];
        for (int j=0; j<num_to_nodes; j++) {
            double tot_currnode = 0;
            for (int k=0; k<num_from_nodes; k++) {
                tot_currnode += curr_from_values[k] * weights[i-1][j][k];
            }
            tot_currnode += biases[i-1][j];
            curr_to_values.push_back(sigmoid(tot_currnode));
        }
        curr_from_values.clear();
        curr_from_values = curr_to_values;
        curr_to_values.clear();
    }

    return curr_from_values;
}


void Network::SGD(
        const traindata_t& training_data, 
        const int& num_epochs,
        const size_t& mini_batch_size,
        const double& eta,
        const testdata_t& test_data) {
    traindata_t training_data_shuffled(training_data);
    size_t n_test = test_data.size();
    size_t n = training_data_shuffled.size();
    for (int j=0; j<num_epochs; j++) {
        std::cout << j << '\n' << std::flush;
        std::shuffle(
                training_data_shuffled.begin(),
                training_data_shuffled.end(),
                generator);
        for (size_t k=0; k<n; k+=mini_batch_size) {
            update_mini_batch(training_data_shuffled.begin()+k,
                              training_data_shuffled.begin()+k+mini_batch_size,
                              eta);
        }
        if (n_test == 0) {
            std::cout << boost::format("Epoch %1% complete\n") % j; 
        }
        else {
            std::cout << boost::format("Epoch %1%: %2% / %3%\n") % j % evaluate(test_data) % n_test;
        }

//            std::vector<std::vector<double>>::const_iterator start = training_data.begin() + k*mini_batch_size;
//            std::vector<std::vector<double>>::const_iterator end = training_data.begin() + (k+1)*mini_batch_size;
//            currbatch = std::vector<std::vector<double>>(start, end);
    }
}


void Network::update_mini_batch(
        const traindata_t::const_iterator& batch_start,
        const traindata_t::const_iterator& batch_end,
        const double& eta) {
    networknodes_t nabla_b;
    networknodes_t delta_nabla_b;
    networkconnections_t nabla_w;
    networkconnections_t delta_nabla_w;

    //Initialize
    layerconnections_t curr_wvec;
    for (size_t i=1; i<num_layers; i++) {
        nabla_b.push_back(layernodes_t(sizes[i], 0));
        for (size_t j=0; j<sizes[i]; j++) {
            curr_wvec.push_back(std::vector<double>(sizes[i-1], 0));
        }
        nabla_w.push_back(curr_wvec);
        curr_wvec.clear();
    }

    //update
    network_t res;
    traindata_t mini_batch(batch_start, batch_end);
    for (size_t i=0; i<mini_batch.size(); i++) {
        res = backprop(std::get<0>(mini_batch[i]),
                       std::get<1>(mini_batch[i]));
        delta_nabla_b = std::get<0>(res);
        delta_nabla_w = std::get<1>(res);
        for (size_t j=1; j<num_layers; j++) {
            for (size_t k=0; k<sizes[j]; k++) {
		nabla_b[j-1][k] += delta_nabla_b[j-1][k];
                for (size_t m=0; m<sizes[j-1]; m++) {
		    nabla_w[j-1][k][m] += delta_nabla_w[j-1][k][m];
                }
            }
        }
        delta_nabla_b.clear();
        delta_nabla_w.clear();
    }


    //Finally update main weights
    float batch_size = float(mini_batch.size());
    for (size_t i=1; i<num_layers; i++) {
        for (size_t j=0; j<sizes[i]; j++) {
	    biases[i-1][j] = biases[i-1][j] - (eta / batch_size) * nabla_b[i-1][j];
            for (size_t k=0; k<sizes[i-1]; k++) {
		weights[i-1][j][k] = weights[i-1][j][k] - (eta / batch_size) * nabla_w[i-1][j][k];
            }
        }
    }
}


network_t Network::backprop(const activations_t& x, const activations_t& y) {
    //We're gonna be filling up these in the wrong order and then reverse at the end
    networknodes_t nabla_b;
    networkconnections_t nabla_w;

    //Important: activations will have the same size as sizes - i.e. one more
    //than the size of weights and biases
    activations_t activation(x);
    std::vector<activations_t> activations;
    activations.push_back(activation);
    std::vector<activations_t> zs;
    activations_t z;
    assert(weights.size() == num_layers-1);
    for (int i=1; i<num_layers; i++) {
        assert(weights[i-1].size() == sizes[i]);
	for (int j=0; j<sizes[i]; j++) {
            assert(weights[i-1][j].size() == sizes[i-1]);
	    double currz = 0;
	    for (int k=0; k<sizes[i-1]; k++) {
		currz += weights[i-1][j][k] * activation[k];
	    }
	    currz += biases[i-1][j];
	    z.push_back(currz);
	}
	activation.clear();
	for (int j=0; j<sizes[i]; j++) {
	    double curr_activation = sigmoid(z[j]);
	    activation.push_back(curr_activation);
	}
	activations.push_back(activation);
	zs.push_back(z);
	z.clear();
    }

    int last_layer = num_layers - 2;
    layernodes_t delta;
    for (int j=0; j<sizes[last_layer+1]; j++) {
	delta.push_back(
                cost_derivative(activations[last_layer+1][j], y[j]) *
                sigmoid_prime(zs[last_layer][j])
                );
    }
    nabla_b.push_back(delta);
    networknodes_t placeholder; 
    layernodes_t placeholder2;
    for (int j=0; j<sizes[last_layer+1]; j++) {
	for (int k=0; k<sizes[last_layer]; k++) {
            placeholder2.push_back(delta[j] * activations[last_layer][k]);
	}
        placeholder.push_back(placeholder2);
        placeholder2.clear();
    }
    nabla_w.push_back(placeholder);
    placeholder.clear();

    layernodes_t updated_delta;
    for (int i=1; i<num_layers-1; i++) {
	z = zs[last_layer-i];
	// Here we want to calculate delta by dotting the current delta with
	// the weights of the previous step, which is why we loop through
	// sizes[last_layer-i+2] instead of sizes[last_layer-i+1]
	// We also want to switch the order of the for loops because we are
	// moving backwards in the network now
	for (int k=0; k<sizes[last_layer-i+1]; k++) {
	    double currdelta = 0;
	    for (int j=0; j<sizes[last_layer-i+2]; j++) {
		currdelta += delta[j] * weights[last_layer-i+1][j][k];
	    }
	    currdelta *= sigmoid_prime(z[k]);
	    updated_delta.push_back(currdelta);
	}
        delta.clear();
        delta = updated_delta;
        updated_delta.clear();
        nabla_b.push_back(delta);
        for (int j=0; j<sizes[last_layer-i+1]; j++) {
            for (int k=0; k<sizes[last_layer-i]; k++) {
                placeholder2.push_back(delta[j] * activations[last_layer-i][k]);
            }
            placeholder.push_back(placeholder2);
            placeholder2.clear();
        }
        nabla_w.push_back(placeholder);
        placeholder.clear();
    }
    std::reverse(nabla_b.begin(), nabla_b.end());
    std::reverse(nabla_w.begin(), nabla_w.end());
    return network_t(nabla_b, nabla_w);
}


double Network::cost_derivative(
        const double& output_activation,
        const double& y) {
    return output_activation - y;
}


int Network::evaluate(const testdata_t& test_data) {
    activations_t x;
    activations_t curr_result;
    int num_correct = 0;
    for (int i=0; i<test_data.size(); i++) {
        x = std::get<0>(test_data[i]);
        int y = std::get<1>(test_data[i]);
        curr_result = feed_forward(x);
        int guessed_num = std::distance(curr_result.begin(),
                std::max_element(
                    curr_result.begin(),
                    curr_result.end()));
        if (guessed_num == y) {
            num_correct += 1;
        }
        x.clear();
    }
    return num_correct;
}
