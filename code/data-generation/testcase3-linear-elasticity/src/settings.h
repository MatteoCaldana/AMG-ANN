#pragma once

#include <array>
#include <string>
#include <vector>

struct Settings {
  const long int dim;
  const long int deg;
  const long int num_refinements;
  const long int cycles;
  const long int renumbering;

  const long int seed;
  const long int mode;
  const long int pattern_size;
  const double max_young;
  const bool sharp;

  const double tol;
  const std::array<double, 3> strong_threshold;

  const bool make_view;
  const long int view_size;

  const bool evaluate_errors;
  const bool output_results;
  const std::string stats_filename;
  const std::string settings_filename;

  const bool output_setup_details;
};

Settings get_settings(const std::string& filename);