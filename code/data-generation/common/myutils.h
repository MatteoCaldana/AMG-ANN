#pragma once

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>

namespace math {
template <typename T>
T pow(T base, long long power) {
  T result = 1;
  while (power > 0) {
    if (power & 1) {
      result *= base;
    }
    base *= base;
    power >>= 1;
  }
  return result;
}

template <typename T>
void describe(const std::vector<T>& v) {
  auto v_copy = v;
  std::sort(v_copy.begin(), v_copy.end());
  std::cout << "min:    " << v_copy[0] << std::endl;
  std::cout << "Q1:     " << v_copy[v.size() / 4] << std::endl;
  std::cout << "median: " << v_copy[v.size() / 2] << std::endl;
  std::cout << "Q3:     " << v_copy[3 * v.size() / 4] << std::endl;
  std::cout << "max:    " << v_copy.back() << std::endl;
  T sum = 0, sum2 = 0;
  for (const auto& e : v) {
    sum += e;
    sum2 += e * e;
  }
  std::cout << "-------------------------------" << std::endl;
  std::cout << "mean:   " << sum / v.size() << std::endl;
  std::cout << "std:    "
            << std::sqrt(sum2 / v.size() - sum * sum / v.size() / v.size())
            << std::endl;
  std::cout << "-------------------------------" << std::endl;
  std::cout << "-------------------------------" << std::endl;
}

template<typename T>
std::vector<T> random_vec(long int seed, long int len, T max) {
  std::vector<T> v(len);
  std::uniform_real_distribution<T> distribution(0.0, max);
  std::default_random_engine generator(seed);
  std::generate(v.begin(), v.end(), [&]() { return distribution(generator); });
  return v;
}}  // namespace math

namespace itertools {

template<typename T>
std::vector<std::string> split(const std::string& s, T delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (item.size()) result.push_back(item);
  }
  return result;
}

template <typename I>
void print(const I& iterable, std::ostream& out = std::cout,
           const char wrapper = '"', const char sep = ',',
           size_t max = 4294967296) {
  size_t i = 1;
  out << wrapper;
  for (const auto& elem : iterable) {
    out << elem;
    if (i >= max) break;
    if (i != iterable.size()) out << sep;
    ++i;
  }
  out << wrapper;
  return;
}

template <typename I, typename O>
std::vector<O> map(const std::vector<I>& iterable,
                   std::function<O(const I&)> mapper) {
  std::vector<O> result;
  for (const auto& el : iterable) result.push_back(mapper(el));
  return result;
}

template <size_t N, typename U, typename V>
std::array<V, N> map_to_array(const std::vector<U>& iterable,
                              std::function<V(U)> mapper) {
  std::array<V, N> result;
  if (iterable.size() != N) {
    std::cout << "Uncompatible dimensions in array map" << iterable.size()
              << " " << N << std::endl;
    std::exit(-1);
  }
  for (size_t i = 0; i < N; ++i) result[i] = mapper(iterable[i]);
  return result;
}

}  // namespace itertools




