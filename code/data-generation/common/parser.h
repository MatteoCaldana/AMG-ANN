#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

#include "myutils.h"

class PETScOutputParser {
 public:
  enum OPTIONS { monitor, monitor_singular_value, monitor_true_residual };
  enum VECTORS {
    preconditioned_residual,
    true_residual,
    normalized_residual,
    min_singular_value,
    max_singular_value,
    average_convergence_factor
  };

  PETScOutputParser(OPTIONS o) : o(o) {}

  const std::string getOption() const { return string_options[o]; }
  size_t parse(const std::string output);
  std::vector<double> get(VECTORS v) const;
  std::string getOutput() const { return parsed_text; }

 private:
  std::string parsed_text;

  size_t parse1(const std::string& s, size_t l);
  size_t parse2(const std::string& s, size_t l);
  size_t parse3(const std::string& s, size_t l);

  const OPTIONS o;

  std::vector<double> prec_res;
  std::vector<double> true_res;
  std::vector<double> norm_res;
  std::vector<double> max_sv;
  std::vector<double> min_sv;
  std::vector<double> acf;

  inline static const std::vector<std::string> string_options = {
      "-ksp_monitor", "-ksp_monitor_singular_value",
      "-ksp_monitor_true_residual"};
};

class BoomerAMGParser {
 public:
  bool parse(const std::string& output);

  const std::vector<double>& get_rows() const { return m_rows; }
  const std::vector<double>& get_nze() const { return m_nze; }
  const std::vector<double>& get_sparsity() const { return m_sparsity; }
  double get_grid() const { return m_grid; }
  double get_operator() const { return m_operator; }
  double get_memory() const { return m_memory; }

 private:
  int m_num_mpi_tasks, m_num_openmp_threads, m_max_lvls, m_num_lvls;
  double m_theta, m_interpolation_truncation_factor, m_max_row_sum;
  std::string m_coarsening_type, m_measures, m_partition, m_interpolation;
  std::vector<double> m_rows, m_nze, m_sparsity;
  double m_grid, m_operator, m_memory;
};

std::vector<double> PETScOutputParser::get(VECTORS v) const {
  switch (v) {
    case preconditioned_residual:
      return prec_res;
    case true_residual: {
      if (o != monitor_true_residual) {
        std::cout << "ERROR: can't return this vector" << std::endl;
        std::exit(-1);
      }
      return true_res;
    }
    case normalized_residual: {
      if (o != monitor_true_residual) {
        std::cout << "ERROR: can't return this vector" << std::endl;
        std::exit(-1);
      }
      return norm_res;
    }
    case min_singular_value:
      if (o != monitor_singular_value) {
        std::cout << "ERROR: can't return this vector" << std::endl;
        std::exit(-1);
      }
      return min_sv;
    case max_singular_value:
      if (o != monitor_singular_value) {
        std::cout << "ERROR: can't return this vector" << std::endl;
        std::exit(-1);
      }
      return max_sv;
    case average_convergence_factor:
      if (o != monitor_true_residual) {
        std::cout << "ERROR: can't return this vector" << std::endl;
        std::exit(-1);
      }
      if (acf.size() != norm_res.size() - 1) {
        std::cout << "ERROR: corrupted vector size" << std::endl;
        std::exit(-1);
      }
      return acf;
    default:
      std::cout << "ERROR: not recognised vector to return" << std::endl;
      std::exit(-1);
  }
}

size_t PETScOutputParser::parse(const std::string output) {
  int lines = 0;
  size_t line_start = 1;
  switch (o) {
    case monitor: {
      while (line_start > 0) {
        line_start = parse1(output, line_start);
        ++lines;
      }
      break;
    }
    case monitor_singular_value: {
      while (line_start > 0) {
        line_start = parse2(output, line_start);
        ++lines;
      }
      break;
    }
    case monitor_true_residual: {
      while (line_start > 0) {
        line_start = parse3(output, line_start);
        ++lines;
      }
      break;
    }
    default:
      std::exit(-1);
  }
  parsed_text = output;
  return lines;
}

size_t PETScOutputParser::parse1(const std::string& s, size_t l) {
  while (s[l] != 'K' && l < s.length()) ++l;
  if (s[l] != 'K') return 0;
  char* end;
  prec_res.push_back(std::strtod(&s[l + 18], &end));
  return l + 38;
}

size_t PETScOutputParser::parse2(const std::string& s, size_t l) {
  while (s[l] != 'K' && l < s.length()) ++l;
  if (s[l] != 'K') return 0;
  char* end;
  prec_res.push_back(std::strtod(&s[l + 18], &end));
  max_sv.push_back(std::strtod(&s[l + 43], &end));
  min_sv.push_back(std::strtod(&s[l + 66], &end));
  return l + 112;
}

size_t PETScOutputParser::parse3(const std::string& s, size_t l) {
  while ((!((s[l] == 'K') && (s[l + 1] == 'S') && (s[l + 2] == 'P') &&
            (s[l + 3] == ' '))) &&
         l < (s.length() - 3))
    ++l;
  if (s[l] != 'K') return 0;
  char* end;
  prec_res.push_back(std::strtod(&s[l + 30], &end));
  true_res.push_back(std::strtod(&s[l + 65], &end));
  norm_res.push_back(std::strtod(&s[l + 99], &end));

  return l + 118;
}

bool BoomerAMGParser::parse(const std::string& output) {
  std::smatch sm;
  auto str = output;

  std::regex re(R"(

 Num MPI tasks = ([0-9]+)

 Num OpenMP threads = ([0-9]+)


BoomerAMG SETUP PARAMETERS:

 Max levels = ([0-9]+)
 Num levels = ([0-9]+)

 Strength Threshold = ([0-9\.]+)
 Interpolation Truncation Factor = ([0-9\.]+)
 Maximum Row Sum Threshold for Dependency Weakening = ([0-9\.]+)

 Coarsening Type = (.*)
(.*\n.*\n.*\n.*\n)? .*


 (.*)

 Interpolation = (.*)

Operator Matrix Information:

 +nonzero +entries/row +row sums
lev +rows +entries +sparse +min +max +avg +min +max
=+
(([ 0-9e\+\.\-]+\n)+)

Interpolation Matrix Information:
 +entries/row +min +max +row sums
lev +rows x cols +min +max +avgW +weight +weight +min +max
=+
(([ 0-9xe\+\.\-]+\n)*)

 +Complexity: +grid = ([0-9\.]+)
 +operator = ([0-9\.]+)
 +memory = ([0-9\.]+).*
)");
  auto search_result = std::regex_search(str, sm, re);
  if (search_result) {
    if (sm.size() == 19) {
      m_num_mpi_tasks = std::stoi(sm.str(1));
      m_num_openmp_threads = std::stoi(sm.str(2));
      m_max_lvls = std::stoi(sm.str(3));
      m_num_lvls = std::stoi(sm.str(4));
      m_theta = std::stod(sm.str(5));
      m_interpolation_truncation_factor = std::stod(sm.str(6));
      m_max_row_sum = std::stod(sm.str(7));
      m_coarsening_type = sm.str(8);
      m_measures = sm.str(9);
      m_partition = sm.str(10);
      m_interpolation = sm.str(11);
      const auto lines = itertools::split(sm.str(12), '\n');
      const auto parsed_lines =
          itertools::map<std::string, std::vector<double>>(
              lines, [](const auto& line) {
                return itertools::map<std::string, double>(
                    itertools::split(line, ' '),
                    [](const auto& x) { return std::stod(x); });
              });
      for (const auto& l : parsed_lines) {
        m_rows.push_back(l[1]);
        m_nze.push_back(l[2]);
        m_sparsity.push_back(l[3]);
      }
      // TODO: parse 14 i.e. Interpolation Matrix Information:
      m_grid = std::stod(sm.str(16));
      m_operator = std::stod(sm.str(17));
      m_memory = std::stod(sm.str(18));
    } else {
      std::cout << "Wrong number of groups: " << sm.size() << std::endl;
      search_result = false;
    }
  } else {
    std::cout << "Do not match re" << std::endl;
    std::cout << str << std::endl;
  }
  return search_result;
}