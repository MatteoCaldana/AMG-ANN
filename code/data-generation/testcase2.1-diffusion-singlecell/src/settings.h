#pragma once

#include <array>
#include <string>

struct Settings {
  const long int dim;
  const long int degree;

  const long int renumbering;
  const double diffusion;
  const long int num_ref;
  const std::array<double, 3> marked_point;
  const double toll;
  const bool hermitian;

  const bool output_results;
  const long int view_size;
  const long int solver_mode;

  const std::string stats_filename;
  const std::string settings_filename;

  const bool output_setup_details;
};

Settings get_settings(const std::string& filename);