#include "utils.h"
#include <iostream>

void print_vec(std::vector<std::vector<std::vector<double>>> vec) {
    for (int i=0; i<vec.size(); i++) {
	for (int j=0; j<vec[i].size(); j++) {
	    for (int k=0; k<vec[i][j].size(); k++) {
		std::cout << vec[i][j][k] << '\n';
	    }
	}
    }
}


void print_vec(std::vector<std::vector<double>> vec) {
    for (int i=0; i<vec.size(); i++) {
	for (int j=0; j<vec[i].size(); j++) {
	    std::cout << vec[i][j] << '\n';
	}
    }
}

void print_vec(std::vector<double> vec) {
    for (int i=0; i<vec.size(); i++) {
	std::cout << vec[i] << '\n';
    }
}
