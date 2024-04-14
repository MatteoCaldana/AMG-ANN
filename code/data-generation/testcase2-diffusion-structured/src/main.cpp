#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <cmath>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>

#include "../../common/amg_solver.h"
#include "../../common/cube_solution.h"
#include "../../common/myutils.h"
#include "../../common/parser.h"
#include "../../common/redirector.h"
#include "../../common/view_maker.h"
#include "settings.h"

namespace LaplacianPETSc {
using namespace dealii;
using namespace std::chrono;
using BoomerAMGData = PETScWrappers::PreconditionBoomerAMG::AdditionalData;

using cube_solution::Solution1D;
using cube_solution::solutions;

class SolutionBase {
 public:
  SolutionBase(int pattern_size, int mode, const std::vector<double>& epsv)
      : m_pattern_size(pattern_size), m_mode(mode), m_epsv(epsv) {
    if (size_t(math::pow(pattern_size, mode)) != m_epsv.size()) {
      std::cout << "Incompatible epsv size: " << m_epsv.size()
                << " should instead be " << math::pow(pattern_size, mode)
                << " (" << pattern_size << "^" << mode << ")" << std::endl;
      std::exit(-1);
    }
    if (m_mode <= 0 || m_mode > 3) {
      std::cout << "Incompatible mode: " << m_mode << std::endl;
      std::exit(-1);
    }
  }

  static int get_sol_id(int pattern_size) {
    return 3 * (1 - (pattern_size % 2));
  }

 protected:
  const int m_pattern_size;          // dimension of the pattern
  const int m_mode;                  // planes, lines or cells (1, 2, 3)
  const std::vector<double> m_epsv;  // diffusion coefficient discontinuity exp

  const double m_h = 2.0 / double(m_pattern_size);  // diff cell side dimension
  const double m_f = M_PI / m_h;                    // frequency of the solution
  const std::vector<double> m_diffv =
      itertools::map<double, double>(m_epsv, [](auto x) {
        return std::pow(10., x);
      });  // diffusion coefficient value 1
  const Solution1D m_sol = solutions[get_sol_id(m_pattern_size)];
};

template <int dim>
class DiffusionCoef : public Function<dim>, public SolutionBase {
 public:
  virtual double value(const Point<dim>& p,
                       const unsigned int component = 0) const override;

  DiffusionCoef(int pattern_size, int mode, const std::vector<double>& epsv)
      : SolutionBase(pattern_size, mode, epsv) {
    if (dim < m_mode) std::exit(-1);
  }
};

template <int dim>
double DiffusionCoef<dim>::value(const Point<dim>& p,
                                 const unsigned int) const {
  long cell_indicator = 0;
  for (int i = 0; i < m_mode; ++i)
    cell_indicator += long(std::trunc((p(i) + 1.) / (m_h + 1e-15))) *
                      math::pow(m_pattern_size, i);
  if (cell_indicator >= long(m_epsv.size()) || cell_indicator < 0) {
    std::cout << "cell_indicator out of range" << std::endl;
    std::exit(-1);
  }
  return m_diffv[cell_indicator];
}

template <int dim>
class Solution : public DiffusionCoef<dim> {
 public:
  virtual double value(const Point<dim>& p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(
      const Point<dim>& p, const unsigned int component = 0) const override;

  using DiffusionCoef<dim>::DiffusionCoef;
};

template <int dim>
double Solution<dim>::value(const Point<dim>& p, const unsigned int) const {
  double return_value = 1.0;
  for (int i = 0; i < dim; ++i)
    return_value *= SolutionBase::m_sol.f[0](p(i), SolutionBase::m_f);
  return return_value;
}

template <int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim>& p,
                                       const unsigned int) const {
  Tensor<1, dim> return_value;
  for (int i = 0; i < dim; ++i) {
    return_value[i] = 1.0;
    for (int j = 0; j < dim; ++j) {
      return_value[i] *= SolutionBase::m_sol.f[i == j](p(j), SolutionBase::m_f);
    }
  }
  return return_value;
}

template <int dim>
class RightHandSide : public DiffusionCoef<dim> {
 public:
  virtual double value(const Point<dim>& p,
                       const unsigned int component = 0) const override;

  using DiffusionCoef<dim>::DiffusionCoef;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim>& p,
                                 const unsigned int) const {
  double return_value = 0;
  for (int i = 0; i < dim; ++i) {
    double double_derivative = 1.0;
    for (int j = 0; j < dim; ++j) {
      double_derivative *=
          SolutionBase::m_sol.f[2 * (i == j)](p(j), SolutionBase::m_f);
    }
    return_value += double_derivative;
  }
  return -return_value;
}

template <int dim>
class Problem {
 public:
  Problem(const Settings& settings);
  void run();

 private:
  void setup_system();
  void assemble_system();
  void output_results() const;
  void process_solution();
  void print_stats(const int cycle);

  const Settings m_settings;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream pcout;

  Triangulation<dim> triangulation;
  FE_Q<dim> fe;
  DoFHandler<dim> dof_handler;

  AffineConstraints<double> hanging_node_constraints;

  PETScWrappers::MPI::SparseMatrix system_matrix;

  PETScWrappers::MPI::Vector solution, m_zero_solution;
  PETScWrappers::MPI::Vector system_rhs;

  ViewMaker m_viewmaker;
  const std::string stats_filename;
  std::fstream filestream;

  std::vector<double> m_error_l2, m_error_h1, m_error_loo;
};

template <int dim>
Problem<dim>::Problem(const Settings& settings)
    : m_settings(settings),
      mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, (this_mpi_process == 0)),
      fe(settings.deg),
      dof_handler(triangulation),
      m_viewmaker(settings.view_size),
      stats_filename(m_settings.stats_filename) {
  if (n_mpi_processes > 1) {
    std::cout << "MPI not supported" << std::endl;
    std::exit(-1);
  }
}

template <int dim>
void Problem<dim>::setup_system() {
  GridTools::partition_triangulation(n_mpi_processes, triangulation);

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::subdomain_wise(dof_handler);

  hanging_node_constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler,
                                          hanging_node_constraints);
  hanging_node_constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, hanging_node_constraints,
                                  false);

  const std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
  const IndexSet locally_owned_dofs =
      locally_owned_dofs_per_proc[this_mpi_process];

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                       mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void Problem<dim>::assemble_system() {
  QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<double> rhs_values(n_q_points), mu_values(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const RightHandSide<dim> right_hand_side(m_settings.pattern_size,
                                           m_settings.mode, m_settings.epsv);
  const DiffusionCoef<dim> diffusion_coef(m_settings.pattern_size,
                                          m_settings.mode, m_settings.epsv);

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    if (cell->subdomain_id() == this_mpi_process) {
      cell_matrix = 0.0;
      cell_rhs = 0.0;

      fe_values.reinit(cell);

      right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);
      diffusion_coef.value_list(fe_values.get_quadrature_points(), mu_values);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            cell_matrix(i, j) +=
                (mu_values[q_point] *                // mu(x_q)
                 fe_values.shape_grad(i, q_point) *  // grad phi_i(x_q)
                 fe_values.shape_grad(j, q_point) *  // grad phi_j(x_q)
                 fe_values.JxW(q_point));            // dx
          }

          cell_rhs(i) += (mu_values[q_point] *
                          fe_values.shape_value(i, q_point) *  // phi_i(x_q)
                          rhs_values[q_point] *                // f(x_q)
                          fe_values.JxW(q_point));             // dx
        }
      }

      cell->get_dof_indices(local_dof_indices);
      hanging_node_constraints.distribute_local_to_global(
          cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(
      dof_handler, 0,
      Solution<dim>(m_settings.pattern_size, m_settings.mode, m_settings.epsv),
      boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                     system_rhs, false);
  m_zero_solution = solution;  // in assemble we just set solution = 0;
}

template <int dim>
void Problem<dim>::output_results() const {
  Vector<double> localized_solution(solution);
  const auto tstamp = std::chrono::system_clock::now();
  const auto tstamp_str = std::to_string(
      duration_cast<microseconds>(tstamp.time_since_epoch()).count());
  std::ofstream output("solution-" + tstamp_str + ".vtk");
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(localized_solution, "solution");
  // export partitioning
  std::vector<unsigned int> partition_int(triangulation.n_active_cells());
  GridTools::get_subdomain_association(triangulation, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");
  // export the mu pattern
  Vector<double> mu_at_dof(solution);
  std::map<types::global_dof_index, Point<dim>> support;
  DoFTools::map_dofs_to_support_points(MappingQGeneric<dim>(fe.degree),
                                       dof_handler, support);
  const DiffusionCoef<dim> mu(m_settings.pattern_size, m_settings.mode,
                              m_settings.epsv);
  for (const auto& [k, v] : support) mu_at_dof[k] = mu.value(v);
  data_out.add_data_vector(mu_at_dof, "mu");
  //
  data_out.build_patches();
  data_out.write_vtk(output);
  return;
}

template <int dim>
void Problem<dim>::process_solution() {
  Vector<float> difference_per_cell(triangulation.n_active_cells());
  const Vector<double> localized_solution(solution);
  const Solution<dim> sol_ex(m_settings.pattern_size, m_settings.mode,
                             m_settings.epsv);

  VectorTools::integrate_difference(
      dof_handler, localized_solution, sol_ex, difference_per_cell,
      QGauss<dim>(fe.degree + 1), VectorTools::L2_norm);
  const double L2_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::L2_norm);

  VectorTools::integrate_difference(
      dof_handler, localized_solution, sol_ex, difference_per_cell,
      QGauss<dim>(fe.degree + 1), VectorTools::H1_seminorm);
  const double H1_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::H1_seminorm);

  const QTrapezoid<1> q_trapez;
  const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
  VectorTools::integrate_difference(dof_handler, localized_solution, sol_ex,
                                    difference_per_cell, q_iterated,
                                    VectorTools::Linfty_norm);
  const double Linfty_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::Linfty_norm);

  std::cout << " L2  Error: " << L2_error << std::endl;
  std::cout << " H1  Error: " << H1_error << std::endl;
  std::cout << " inf Error: " << Linfty_error << std::endl;

  m_error_l2.push_back(L2_error);
  m_error_h1.push_back(H1_error);
  m_error_loo.push_back(Linfty_error);
}

void compute_order(const std::vector<double>& v, double ref_order) {
  bool ok = true;
  std::cout << std::fixed << std::showpoint;
  std::cout << std::setprecision(4);
  for (size_t i = 0; i < v.size() - 1; ++i) {
    const auto order = std::log2(v[i]) - std::log2(v[i + 1]);
    ok &= (order - ref_order > -0.2);
    std::cout << order << ", ";
  }
  std::cout << (ok ? "OK" : "KO") << " vs " << int(ref_order) << std::endl;
}

template <int dim>
void Problem<dim>::run() {
  filestream.open(stats_filename, std::fstream::out | std::fstream::app);
  if (m_settings.make_view) {
    filestream
        << "setting,dim,ndof,mesh_ref,degree,sol_id,sol_pattern_size,epsv,"
           "mode,timestamp,t_view,view_size,view,view_count,view_max_pp,"
           "view_max_np\n";
  } else {
    filestream
        << "setting,dim,ndof,mesh_ref,degree,sol_id,sol_pattern_size,epsv,mode,"
           "timestamp,theta,maxrowsum,symop,tol,t_amg_setup,";
    if (m_settings.output_setup_details) {
      filestream << "nrows,nze,sparsity,grid,operator,memory,";
    }
    filestream << "t_solve,niters,p_res\n";
  }

  std::vector<double> elapsed_time;
  for (int cycle = 0; cycle < m_settings.cycles; ++cycle) {
    const auto t0 = high_resolution_clock::now();
    pcout << "Cycle " << cycle << ':' << std::endl;
    if (cycle == 0) {
      GridGenerator::subdivided_hyper_cube(triangulation,
                                           m_settings.pattern_size, -1, 1);
      triangulation.refine_global(m_settings.num_refinements);
    } else {
      triangulation.refine_global(1);
    }

    setup_system();
    pcout << "Assembling system" << std::endl;
    assemble_system();

    if (m_settings.make_view) {
      print_stats(cycle);
      pcout << "Making view" << std::endl;
      m_viewmaker.make_view(filestream, system_matrix);
      m_viewmaker.print_view(filestream);
    } else {
      const auto& thetav = m_settings.strong_threshold;
      const auto& mrsv = m_settings.max_row_sum;
      const auto& symv = m_settings.symmetric_operator;
      for (double t = thetav[0]; t <= thetav[1]; t += thetav[2]) {
        for (double mrs = mrsv[0]; mrs <= mrsv[1]; mrs += mrsv[2]) {
          for (int sym = symv[0]; sym <= symv[1]; sym++) {
            solution = m_zero_solution;  // set first iteration x^{0} = 0.0
            const BoomerAMGData data(
                sym,  //  symmetric_operator=false,
                t,    //  strong_threshold=0.25
                mrs,  //  max_row_sum=0.9
                0,    //  aggressive_coarsening_num_levels=0
                m_settings.output_setup_details  //  output_details=false
            );
            print_stats(cycle);
            pcout << "Solving system" << std::endl;
            const auto n_iterations = amg_solver::amg_solve(
                data, m_settings.tol, filestream, system_matrix, system_rhs,
                solution, mpi_communicator, stats_filename,
                hanging_node_constraints);  // solve(data);
            pcout << "Solver iterations: " << n_iterations << std::endl;
            pcout
                << "========================================================\n"
                   "========================================================\n"
                   "========================================================\n";
          }
        }
      }
      const auto t1 = high_resolution_clock::now();
      elapsed_time.push_back(duration_cast<microseconds>(t1 - t0).count() /
                             1000.0);
      if (m_settings.output_results) {
        pcout << "Exporting results" << std::endl;
        output_results();
      }
      if (m_settings.evaluate_errors) {
        pcout << "Evaluating errors" << std::endl;
        process_solution();
      }
    }
    pcout << "============================================================\n"
             "============================================================\n"
             "============================================================\n"
             "============================================================\n"
             "============================================================\n";
  }
  if (m_settings.evaluate_errors) {
    std::cout << "## " << m_settings.pattern_size << " " << m_settings.mode
              << " ";
    itertools::print(elapsed_time);
    std::cout << std::endl;
    compute_order(m_error_l2, fe.degree + 1);
    compute_order(m_error_h1, fe.degree);
    compute_order(m_error_loo, fe.degree + 1);
  }
  filestream.close();
}

template <int dim>
void Problem<dim>::print_stats(const int cycle) {
  pcout << " Active cells:   " << triangulation.n_active_cells() << std::endl;
  pcout << " Number of dofs: " << dof_handler.n_dofs() << std::endl;
  std::time_t timestamp = std::time(nullptr);
  filestream << std::scientific << std::setprecision(17);
  filestream << m_settings.settings_filename << "," << dim << ","
             << dof_handler.n_dofs() << ","
             << m_settings.num_refinements + cycle << "," << fe.degree << ","
             << SolutionBase::get_sol_id(m_settings.pattern_size) << ","
             << m_settings.pattern_size << ",";
  itertools::print(m_settings.epsv, filestream);
  filestream << "," << m_settings.mode << "," << timestamp << ",";

  pcout << "\nepsv: ";
  itertools::print(m_settings.epsv);
  pcout << "\ntimestamp: " << timestamp
        << "\npattern size: " << m_settings.pattern_size
        << "\nmode: " << m_settings.mode << std::endl;
}

}  // namespace LaplacianPETSc

int main(int argc, char** argv) {
  using namespace dealii;
  using namespace LaplacianPETSc;
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const auto settings =
        get_settings(argc > 1 ? std::string(argv[1]) : "../settings.json");

    if (settings.dim == 2) {
      Problem<2> problem(settings);
      problem.run();
    } else if (settings.dim == 3) {
      Problem<3> problem(settings);
      problem.run();
    } else {
      std::cout << "Unhandled dimesion: " << settings.dim << std::endl;
      std::exit(-1);
    }

  } catch (std::exception& exc) {
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl;
    return 1;
  }
  return 0;
}