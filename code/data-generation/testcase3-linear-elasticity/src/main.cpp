#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <chrono>
#include <fstream>
#include <iostream>

#include "../../common/amg_solver.h"
#include "../../common/myutils.h"
#include "../../common/view_maker.h"
#include "settings.h"

namespace MyLinearElasticity {
using namespace dealii;
using BoomerAMGData = PETScWrappers::PreconditionBoomerAMG::AdditionalData;
constexpr double poisson_ratio = 0.29;
constexpr double min_young_module = 1000.0;  // minimum young modulus

void right_hand_side(const std::vector<Point<3>>& points,
                     std::vector<Tensor<1, 3>>& values,
                     const std::vector<double>& mu_values,
                     const std::vector<double>& lambda_values,
                     long int pattern_size) {
  AssertDimension(values.size(), points.size());
  AssertDimension(values.size(), mu_values.size());
  AssertDimension(values.size(), lambda_values.size());

  const double pi = numbers::PI * pattern_size / 2.0;
  const double pi2 = pi * pi;

  for (size_t i = 0; i < points.size(); ++i) {
    for (int component = 0; component < 3; ++component) {
      const double x = points[i][(0 + component) % 3];
      const double y = points[i][(1 + component) % 3];
      const double z = points[i][(2 + component) % 3];

      const double siny = std::sin(pi * y);
      const double sinz = std::sin(pi * z);

      values[i][component] =
          2 * pi2 *
          (-0.25 * lambda_values[i] *
               (std::cos(pi * (-2 * x + y + z)) +
                std::cos(pi * (2 * x - y + z)) +
                std::cos(pi * (2 * x + y - z)) -
                3 * std::cos(pi * (2 * x + y + z))) *
               siny * sinz -
           mu_values[i] *
               (std::sin(pi * x) * siny * siny * std::sin(pi * (x + 2 * z)) +
                std::sin(pi * x) * sinz * sinz * std::sin(pi * (x + 2 * y)) +
                2 * siny * siny * sinz * sinz * std::cos(2 * pi * x)));
    }
  }
}

void mu(const std::vector<Point<3>>& points, std::vector<double>& values,
        const std::vector<double>& epsv, long int mode, long int pattern_size) {
  AssertDimension(values.size(), points.size());
  const auto h = 2.0 / static_cast<double>(pattern_size) + 1e-15;
  for (size_t k = 0; k < points.size(); ++k) {
    long cell_indicator = 0;
    for (int i = 0; i < mode; ++i)
      cell_indicator += static_cast<long>(std::trunc((points[k](i) + 1.) / h)) *
                        math::pow(pattern_size, i);
    AssertIndexRange(cell_indicator, epsv.size());
    values[k] = min_young_module * epsv[cell_indicator] / (1.0 + poisson_ratio);
  }
}

void lambda(const std::vector<Point<3>>& points, std::vector<double>& values,
            const std::vector<double>& epsv, long int mode,
            long int pattern_size) {
  AssertDimension(values.size(), points.size());
  const auto beta = poisson_ratio / (1.0 - 2.0 * poisson_ratio);
  mu(points, values, epsv, mode, pattern_size);
  for (auto& v : values) v *= beta;
}

class ExactSolution : public Function<3> {
 public:
  ExactSolution(long int pattern_size)
      : Function<3>(3), m_pattern_size(pattern_size) {}

  virtual double value(const Point<3>& p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, 3, double> gradient(
      const Point<3>& p, const unsigned int component = 0) const override;

 private:
  const long int m_pattern_size;
};

double ExactSolution::value(const Point<3>& p, const unsigned int) const {
  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  const double pi = numbers::PI * m_pattern_size / 2.0;
  const auto partial = std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
  return partial * partial;
}

Tensor<1, 3, double> ExactSolution::gradient(const Point<3>& p,
                                             const unsigned int) const {
  const double x = p[0];
  const double y = p[1];
  const double z = p[2];

  const double pi = numbers::PI * m_pattern_size / 2.0;

  const auto partial = std::sin(pi * x) * std::sin(pi * y) * std::sin(pi * z);
  return Tensor<1, 3, double>{
      {2.0 * pi * partial * partial / std::tan(pi * x),
       2.0 * pi * partial * partial / std::tan(pi * y),
       2.0 * pi * partial * partial / std::tan(pi * z)}};
}

template <int dim>
class ElasticProblem {
 public:
  ElasticProblem(const Settings& settings);
  void run();

 private:
  void setup_system();
  void assemble_system();
  void compute_errors() const;
  void output_results(const unsigned int cycle) const;
  void print_stats(const unsigned int cycle);

  const Settings m_settings;

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;

  FESystem<dim> fe;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;
  PETScWrappers::MPI::SparseMatrix system_matrix;

  PETScWrappers::MPI::Vector zero, solution;
  PETScWrappers::MPI::Vector system_rhs;

  const std::vector<double> m_epsv;

  mutable ConvergenceTable m_convergence_table;

  ViewMaker m_viewmaker;
  const std::string m_stats_filename;
  std::fstream m_filestream;
};

template <int dim>
ElasticProblem<dim>::ElasticProblem(const Settings& settings)
    : m_settings(settings),
      mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      dof_handler(triangulation),
      fe(FE_Q<dim>(m_settings.deg), dim),
      m_epsv(itertools::map<double, double>(
          math::random_vec(m_settings.seed,
                           math::pow(m_settings.pattern_size, m_settings.mode),
                           m_settings.max_young),
          [this](auto x) {
            return std::pow(10.0, m_settings.sharp
                                      ? m_settings.max_young *
                                            (x < m_settings.max_young / 2.0)
                                      : x);
          })),
      m_viewmaker(settings.view_size),
      m_stats_filename(m_settings.stats_filename) {
  if (n_mpi_processes > 1) {
    std::cout << "MPI not supported" << std::endl;
    std::exit(-1);
  }
  std::cout << "-------- epsv --------" << std::endl;
  math::describe(m_epsv);
}

template <int dim>
void ElasticProblem<dim>::print_stats(const unsigned int cycle) {
  std::cout << " Active cells:   " << triangulation.n_active_cells()
            << std::endl;
  std::cout << " Number of dofs: " << dof_handler.n_dofs() << std::endl;
  std::time_t timestamp = std::time(nullptr);
  m_filestream << std::scientific << std::setprecision(17);
  m_filestream << m_settings.settings_filename << "," << dim << ","
               << dof_handler.n_dofs() << ","
               << m_settings.num_refinements + cycle << "," << fe.degree << ","
               << m_settings.seed << "," << m_settings.mode << ","
               << m_settings.pattern_size << "," << m_settings.max_young << ","
               << m_settings.sharp << "," << m_settings.renumbering << ","
               << timestamp << ",";

  std::cout << "\nseed: " << m_settings.seed << "\ntimestamp: " << timestamp
            << "\npattern size: " << m_settings.pattern_size
            << "\nmode: " << m_settings.mode << std::endl;
}

template <int dim>
void ElasticProblem<dim>::setup_system() {
  GridTools::partition_triangulation(n_mpi_processes, triangulation);

  dof_handler.distribute_dofs(fe);
  switch (m_settings.renumbering) {
    case 0:
      // default, equivalent to subdomain_wise
      break;
    case 1:
      DoFRenumbering::Cuthill_McKee(dof_handler);
      break;
    case 2:
      DoFRenumbering::boost::king_ordering(dof_handler);
      break;
    case 3:
      DoFRenumbering::boost::Cuthill_McKee(dof_handler);
      break;
    default:
      std::cout << "ERROR: Unknown renumbering ID" << std::endl;
      std::exit(-1);
      break;
  }

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(
      dof_handler, 0, ExactSolution(m_settings.pattern_size), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,
                                  /*keep_constrained_dofs = */ false);

  const std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
  const IndexSet locally_owned_dofs =
      locally_owned_dofs_per_proc[this_mpi_process];

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                       mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  zero.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void ElasticProblem<dim>::assemble_system() {
  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> lambda_values(n_q_points);
  std::vector<double> mu_values(n_q_points);
  std::vector<Tensor<1, dim>> rhs_values(n_q_points);

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit(cell);

    mu(fe_values.get_quadrature_points(), mu_values, m_epsv, m_settings.mode,
       m_settings.pattern_size);
    lambda(fe_values.get_quadrature_points(), lambda_values, m_epsv,
           m_settings.mode, m_settings.pattern_size);
    right_hand_side(fe_values.get_quadrature_points(), rhs_values, mu_values,
                    lambda_values, m_settings.pattern_size);

    for (const unsigned int i : fe_values.dof_indices()) {
      const unsigned int component_i = fe.system_to_component_index(i).first;

      for (const unsigned int j : fe_values.dof_indices()) {
        const unsigned int component_j = fe.system_to_component_index(j).first;

        for (const unsigned int q_point :
             fe_values.quadrature_point_indices()) {
          cell_matrix(i, j) +=
              ((fe_values.shape_grad(i, q_point)[component_i] *
                fe_values.shape_grad(j, q_point)[component_j] *
                lambda_values[q_point]) +
               (fe_values.shape_grad(i, q_point)[component_j] *
                fe_values.shape_grad(j, q_point)[component_i] *
                mu_values[q_point]) +
               ((component_i == component_j)
                    ? (fe_values.shape_grad(i, q_point) *
                       fe_values.shape_grad(j, q_point) * mu_values[q_point])
                    : 0)) *
              fe_values.JxW(q_point);
        }
      }
    }

    for (const unsigned int i : fe_values.dof_indices()) {
      const unsigned int component_i = fe.system_to_component_index(i).first;

      for (const unsigned int q_point : fe_values.quadrature_point_indices())
        cell_rhs(i) += fe_values.shape_value(i, q_point) *
                       rhs_values[q_point][component_i] *
                       fe_values.JxW(q_point);
    }

    cell->get_dof_indices(local_dof_indices);
    constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void ElasticProblem<dim>::output_results(const unsigned int cycle) const {
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names;
  for (int i = 0; i < dim; ++i) {
    solution_names.push_back(std::string(1, 'x' + i) + "_displacement");
  }
  data_out.add_data_vector(solution, solution_names);
  // export the mu pattern
  Vector<double> mu_at_dof(solution);
  std::map<types::global_dof_index, Point<dim>> support;
  DoFTools::map_dofs_to_support_points(MappingQGeneric<dim>(fe.degree),
                                       dof_handler, support);
  for (const auto& [k, v] : support) {
    std::vector<double> mu_value(1);
    mu({v}, mu_value, m_epsv, m_settings.mode, m_settings.pattern_size);
    mu_at_dof[k] = mu_value[0];
  }
  data_out.add_data_vector(mu_at_dof, "mu");
  //
  data_out.build_patches();

  std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
  data_out.write_vtk(output);
}

template <int dim>
void ElasticProblem<dim>::compute_errors() const {
  Vector<double> difference_per_cell(triangulation.n_active_cells());
  VectorTools::integrate_difference(
      dof_handler, solution, ExactSolution(m_settings.pattern_size),
      difference_per_cell, QGauss<dim>(fe.degree + 1), VectorTools::L2_norm);

  const double L2_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::L2_norm);

  VectorTools::integrate_difference(
      dof_handler, solution, ExactSolution(m_settings.pattern_size),
      difference_per_cell, QGauss<dim>(fe.degree + 1), VectorTools::H1_norm);

  const double H1_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::H1_norm);

  std::cout << "   L2 Error: " << L2_error << std::endl
            << "   H1 Error: " << H1_error << std::endl;

  m_convergence_table.add_value("cells", triangulation.n_active_cells());
  m_convergence_table.add_value("dofs", dof_handler.n_dofs());
  m_convergence_table.add_value("L2", L2_error);
  m_convergence_table.add_value("H1", H1_error);
}

template <int dim>
void ElasticProblem<dim>::run() {
  m_filestream.open(m_stats_filename, std::fstream::out | std::fstream::app);
  if (m_settings.make_view) {
    m_filestream
        << "setting,dim,ndof,mesh_ref,degree,seed,mode,pattern_size,"
           "max_young,sharp,renumbering,timestamp,t_view,view_size,view,view_"
           "count,view_max_pp,view_max_np\n";
  } else {
    m_filestream
        << "setting,dim,ndof,mesh_ref,degree,seed,mode,pattern_size,"
           "max_young,sharp,renumbering,timestamp,theta,maxrowsum,symop,"
           "aggressive_lvls,tol,t_amg_setup,";
    if (m_settings.output_setup_details) {
      m_filestream << "nrows,nze,sparsity,grid,operator,memory,";
    }
    m_filestream << "t_solve,niters,p_res\n";
  }

  for (unsigned int cycle = 0; cycle < m_settings.cycles; ++cycle) {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0) {
      GridGenerator::hyper_cube(triangulation, -1, 1);
      triangulation.refine_global(m_settings.num_refinements);
    } else {
      triangulation.refine_global(1);
    }

    setup_system();
    assemble_system();

    if (m_settings.make_view) {
      print_stats(cycle);
      std::cout << "Making view" << std::endl;
      m_viewmaker.make_view(m_filestream, system_matrix);
      m_viewmaker.print_view(m_filestream);
    } else {
      const auto& thetav = m_settings.strong_threshold;
      for (double t = thetav[0]; t <= thetav[1]; t += thetav[2]) {
        for (int aggr_lvls : {2}) {
          solution = zero;  // set first iteration x^{0} = 0.0
          const BoomerAMGData data(
              true,       //  symmetric_operator=false,
              t,          //  strong_threshold=0.25
              0.9,        //  max_row_sum=0.9
              aggr_lvls,  //  aggressive_coarsening_num_levels=0
              m_settings.output_setup_details  //  output_details=false
          );
          print_stats(cycle);
          std::cout << "Solving system" << std::endl;
          const auto n_iterations = amg_solver::amg_solve(
              data, m_settings.tol, m_filestream, system_matrix, system_rhs,
              solution, mpi_communicator, m_stats_filename, constraints);
          std::cout << "Solver iterations: " << n_iterations << std::endl;
          std::cout << "====================================" << std::endl;
        }
      }

      if (m_settings.output_results) {
        std::cout << "Exporting results" << std::endl;
        output_results(cycle);
      }
      if (m_settings.evaluate_errors) {
        std::cout << "Evaluating errors" << std::endl;
        compute_errors();
      }
      std::cout << "====================================" << std::endl;
      std::cout << "====================================" << std::endl;
      m_filestream << std::flush;
    }
  }
  if (m_settings.evaluate_errors) {
    m_convergence_table.set_scientific("L2", true);
    m_convergence_table.set_scientific("H1", true);
    m_convergence_table.evaluate_convergence_rates(
        "L2", ConvergenceTable::reduction_rate_log2);
    m_convergence_table.evaluate_convergence_rates(
        "H1", ConvergenceTable::reduction_rate_log2);
    std::cout << std::endl;
    m_convergence_table.write_text(std::cout);
  }
}

}  // namespace MyLinearElasticity

int main(int argc, char* argv[]) {
  try {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const auto settings =
        get_settings(argc > 1 ? std::string(argv[1]) : "../settings.json");

    if (settings.dim == 3) {
      MyLinearElasticity::ElasticProblem<3> problem(settings);
      problem.run();
    } else {
      std::cout << "Unhandled dimesion: " << settings.dim << std::endl;
      std::exit(-1);
    }

  } catch (std::exception& exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  } catch (...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}