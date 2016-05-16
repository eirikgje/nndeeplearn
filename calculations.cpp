#include <cmath>

double sigmoid(const double& x) {
    return 1.0 / (1.0 + exp(-x));
}


double sigmoid_prime(const double& z) {
    return sigmoid(z) * (1 - sigmoid(z));
}
