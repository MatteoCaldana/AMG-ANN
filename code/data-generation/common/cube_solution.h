#pragma once

#include <cmath>

namespace cube_solution {
struct Solution0 {
  static double fff(double t, double f) { return std::sin(f * t); }
  static double ffp(double t, double f) { return f * std::cos(f * t); }
  static double fpp(double t, double f) { return -f * f * std::sin(f * t); }
};

struct Solution1 {
  static double fff(double t, double f) {
    return std::sin(f * t) * std::sin(f * t);
  }
  static double ffp(double t, double f) {
    return 2.0 * f * std::cos(f * t) * std::sin(f * t);
  }
  static double fpp(double t, double f) {
    return 2.0 * f * f * std::cos(2.0 * f * t);
  }
};

struct Solution3 {
  static double fff(double t, double f) { return std::cos(f * t); }
  static double ffp(double t, double f) { return -f * std::sin(f * t); }
  static double fpp(double t, double f) { return -f * f * std::cos(f * t); }
};

typedef double (*fun_t)(double, double);  // type of a function
struct Solution1D {
  fun_t f[3];
};  // type of a solution: fff, ffp, fpp
constexpr Solution1D solutions[4] = {
    {{&Solution0::fff, &Solution0::ffp, &Solution0::fpp}},
    {{&Solution1::fff, &Solution1::ffp, &Solution1::fpp}},
    {{nullptr, nullptr, nullptr}},
    {{&Solution3::fff, &Solution3::ffp, &Solution3::fpp}}};
}  // namespace cube_solution