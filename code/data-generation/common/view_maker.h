#pragma once

#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>

#include <chrono>
#include <vector>

#include "myutils.h"

namespace {
using namespace dealii;
using namespace std::chrono;

class ViewMaker {
 public:
  explicit ViewMaker(PetscInt vs)
      : m_view_size(vs),
        system_matrix_view(vs * vs),
        system_matrix_view_max_pp(vs * vs),
        system_matrix_view_max_np(vs * vs),
        system_matrix_view_count(vs * vs) {}

  void make_view(std::fstream& filestream,
                 PETScWrappers::MPI::SparseMatrix& system_matrix) {
    std::fill(system_matrix_view.begin(), system_matrix_view.end(), 0.0);
    std::fill(system_matrix_view_max_pp.begin(),
              system_matrix_view_max_pp.end(), 0.0);
    std::fill(system_matrix_view_max_np.begin(),
              system_matrix_view_max_np.end(), 0.0);
    std::fill(system_matrix_view_count.begin(), system_matrix_view_count.end(),
              0);
    auto t1 = high_resolution_clock::now();
    const auto& petsc_matrix = system_matrix.petsc_matrix();
    PetscInt ncols;
    const PetscInt* cols;
    const PetscScalar* vals;

    const PetscInt n = system_matrix.m();
    const PetscInt q = n / m_view_size;
    const PetscInt q1 = q + 1;
    const PetscInt p = n % m_view_size;
    const PetscInt t = q1 * p;

    for (PetscInt i = 0; i < n; ++i) {
      if (MatGetRow(petsc_matrix, i, &ncols, &cols, &vals) != 0) {
        std::cout << "MatGetRow ierr" << std::endl;
        std::exit(1);
      }
      const auto bin_row = (i < t) ? (i / q1) : ((i - t) / q + p);
      for (PetscInt j = 0; j < ncols; ++j) {
        const PetscInt col = cols[j];
        const auto bin_col = (col < t) ? (col / q1) : ((col - t) / q + p);
        const auto flat_bin = m_view_size * bin_row + bin_col;
        system_matrix_view[flat_bin] += vals[j];
        system_matrix_view_count[flat_bin] += 1;

        system_matrix_view_max_pp[flat_bin] =
            std::max(std::max(vals[j], PetscScalar(0.0)),
                     system_matrix_view_max_pp[flat_bin]);
        system_matrix_view_max_np[flat_bin] =
            std::max(std::max(-vals[j], PetscScalar(0.0)),
                     system_matrix_view_max_np[flat_bin]);
      }
      if (MatRestoreRow(petsc_matrix, i, &ncols, &cols, &vals) != 0) {
        std::cout << "MatRestoreRow ierr" << std::endl;
        std::exit(1);
      }
    }
    auto t2 = high_resolution_clock::now();
    filestream << duration_cast<microseconds>(t2 - t1).count() << ",";
  }
  void print_view(std::fstream& filestream) const {
    filestream << m_view_size << ",";
    itertools::print(system_matrix_view, filestream);
    filestream << ",";
    itertools::print(system_matrix_view_count, filestream);
    filestream << ",";
    itertools::print(system_matrix_view_max_pp, filestream);
    filestream << ",";
    itertools::print(system_matrix_view_max_np, filestream);
    filestream << "\n";
  }

 private:
  const PetscInt m_view_size;
  std::vector<PetscScalar> system_matrix_view, system_matrix_view_max_pp,
      system_matrix_view_max_np;
  std::vector<PetscInt> system_matrix_view_count;
};
}  // namespace