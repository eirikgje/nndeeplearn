#ifndef VECTOROPS_H
#define VECTOROPS_H
#include <vector>


template <typename T>
std::vector<T>& operator+(const std::vector<T>& a, const std::vector<T>& b);

template <typename T>
std::vector<T>& operator*(const double& a, const std::vector<T>& b);

template <typename T>
std::vector<T>& operator-(const std::vector<T>& a, const std::vector<T>& b);
#endif
