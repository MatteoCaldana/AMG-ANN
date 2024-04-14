#pragma once

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <chrono>
#include <fstream>

#include "myutils.h"
#include "parser.h"
#include "redirector.h"

namespace amg_solver {
using namespace dealii;
using namespace std::chrono;
using BoomerAMGData = PETScWrappers::PreconditionBoomerAMG::AdditionalData;

unsigned int amg_solve(const BoomerAMGData& data, double rtol,
                       std::fstream& filestream,
                       PETScWrappers::MPI::SparseMatrix& system_matrix,
                       PETScWrappers::MPI::Vector& system_rhs,
                       PETScWrappers::MPI::Vector& solution,
                       MPI_Comm& mpi_communicator,
                       const std::string& stats_filename,
                       AffineConstraints<double>& hanging_node_constraints) {
  filestream << data.strong_threshold << "," << data.max_row_sum << ","
             << data.symmetric_operator << ","
             << data.aggressive_coarsening_num_levels << "," << rtol << ",";
  SolverControl solver_control(solution.size(), rtol);

  PETScOutputParser parser(PETScOutputParser::monitor);
  PETScWrappers::set_option_value(parser.getOption(), "");

  PETScWrappers::SolverCG cg(solver_control, mpi_communicator);
  PETScWrappers::PreconditionBoomerAMG preconditioner;

  // WARNING!!! using PIPE old redirected output stays in cache causing problems
  CStdoutRedirector redirect(CStdoutRedirector::FILE, stats_filename);
  redirect.init();
  redirect.start();

  {
    auto t1 = high_resolution_clock::now();
    preconditioner.initialize(system_matrix, data);
    auto t2 = high_resolution_clock::now();
    filestream << duration_cast<microseconds>(t2 - t1).count() << ",";
  }

  auto t3 = high_resolution_clock::now();
  cg.solve(system_matrix, solution, system_rhs, preconditioner);
  auto t4 = high_resolution_clock::now();
  filestream << duration_cast<microseconds>(t4 - t3).count() << ",";
  fflush(stdout);
  std::cout << std::endl;
  redirect.stop();

  const auto output = redirect.get();
  if (data.output_details) {
    BoomerAMGParser setup_parser;
    const auto r = setup_parser.parse(output);
    if (!r) {
      std::cout << "Error parsing BoomerAMG setup output" << std::endl;
      std::cout << output << std::endl;
      std::exit(-1);
    }

    itertools::print(setup_parser.get_rows(), filestream);
    filestream << ",";
    itertools::print(setup_parser.get_nze(), filestream);
    filestream << ",";
    itertools::print(setup_parser.get_sparsity(), filestream);
    filestream << ",";
    filestream << setup_parser.get_grid() << "," << setup_parser.get_operator()
               << "," << setup_parser.get_memory() << ",";
  }
  parser.parse(output);
  // hp: the solver gives the residual at iteration 0, i.e. before doing
  // anything
  const auto pr = parser.get(PETScOutputParser::preconditioned_residual);
  filestream << solver_control.last_step() << ",";
  itertools::print(pr, filestream);
  filestream << "\n";

  Vector<double> localized_solution(solution);
  hanging_node_constraints.distribute(localized_solution);
  solution = localized_solution;
  return solver_control.last_step();
}

}  // namespace amg_solver
