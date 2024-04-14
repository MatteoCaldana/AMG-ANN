#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>
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

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <random>

#include "../../common/amg_solver.h"
#include "../../common/cube_solution.h"
#include "../../common/view_maker.h"
#include "settings.h"

namespace MyStructuredDiffusion {
using namespace dealii;
using BoomerAMGData = PETScWrappers::PreconditionBoomerAMG::AdditionalData;

using cube_solution::Solution1D;
using cube_solution::solutions;

template <int dim>
class BaseSolution : public Function<dim> {
 public:
  BaseSolution(size_t id, double f) : m_id(id), m_f(f) {}

 protected:
  const size_t m_id;
  const double m_f;
};

template <int dim>
class Solution : public BaseSolution<dim> {
 public:
  using BaseSolution<dim>::BaseSolution;
  virtual double value(const Point<dim>& p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(
      const Point<dim>& p, const unsigned int component = 0) const override;
};

template <int dim>
double Solution<dim>::value(const Point<dim>& p, const unsigned int) const {
  double return_value = 1.0;
  for (int i = 0; i < dim; ++i)
    return_value *= solutions[this->m_id].f[0](p(i), this->m_f);
  return return_value;
}

template <int dim>
Tensor<1, dim> Solution<dim>::gradient(const Point<dim>& p,
                                       const unsigned int) const {
  Tensor<1, dim> return_value;
  for (int i = 0; i < dim; ++i) {
    return_value[i] = 1.0;
    for (int j = 0; j < dim; ++j) {
      return_value[i] *= solutions[this->m_id].f[i == j](p(j), this->m_f);
    }
  }
  return return_value;
}

template <int dim>
class RightHandSide : public BaseSolution<dim> {
 public:
  using BaseSolution<dim>::BaseSolution;
  virtual double value(const Point<dim>& p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double RightHandSide<dim>::value(const Point<dim>& p,
                                 const unsigned int) const {
  double return_value = 0;
  for (int i = 0; i < dim; ++i) {
    double double_derivative = 1.0;
    for (int j = 0; j < dim; ++j) {
      double_derivative *=
          solutions[this->m_id].f[2 * (i == j)](p(j), this->m_f);
    }
    return_value += double_derivative;
  }
  return -return_value;
}

template <int dim>
class PoissonProblem {
 public:
  PoissonProblem(const Settings& s);

  void run();

 private:
  void setup_system();
  void assemble_system();
  void refine_grid(unsigned int cycle);
  void process_solution(unsigned int cycle);
  void output_solution(unsigned int cycle);
  void print_convergence_table();
  void output_stats(unsigned int cycle);

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  FE_Q<dim> fe;

  AffineConstraints<double> hanging_node_constraints;

  PETScWrappers::MPI::SparseMatrix system_matrix;

  PETScWrappers::MPI::Vector zero_solution, solution;
  PETScWrappers::MPI::Vector system_rhs;
  Vector<double> diffusion_at_dof;

  ConvergenceTable convergence_table;

  std::vector<double> epsv;
  Settings m_settings;

  ViewMaker m_viewmaker;
  const std::string stats_filename;
  std::fstream filestream;
};

template <int dim>
PoissonProblem<dim>::PoissonProblem(const Settings& settings)
    : mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      dof_handler(triangulation),
      fe(settings.degree),
      m_settings(settings),
      m_viewmaker(settings.view_size),
      stats_filename(m_settings.stats_filename) {}

template <int dim>
void PoissonProblem<dim>::setup_system() {
  GridTools::partition_triangulation(n_mpi_processes, triangulation);

  dof_handler.distribute_dofs(fe);
  switch (m_settings.renumbering) {
    case 0:
      DoFRenumbering::subdomain_wise(dof_handler);
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
      break;
  }

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
  zero_solution.reinit(locally_owned_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  diffusion_at_dof.reinit(dof_handler.n_dofs());
}

template <int dim>
void PoissonProblem<dim>::assemble_system() {
  std::cout << "Assembling" << std::endl;
  QGauss<dim> quadrature_formula(fe.degree + 1);

  const unsigned int n_q_points = quadrature_formula.size();
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const RightHandSide<dim> right_hand_side(m_settings.sol_id,
                                           m_settings.sol_freq);
  std::vector<double> rhs_values(n_q_points);

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0.;
    cell_rhs = 0.;

    fe_values.reinit(cell);
    auto ancestor = cell;
    for (int i = 0; i < cell->level() - m_settings.num_bas_ref; ++i)
      ancestor = ancestor->parent();
    const auto eps = epsv[ancestor->index()];
    const auto mu = std::pow(10, eps);

    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          cell_matrix(i, j) +=
              ((mu * fe_values.shape_grad(i, q_point) *  // grad phi_i(x_q)
                fe_values.shape_grad(j, q_point)) *      // grad phi_j(x_q)
               fe_values.JxW(q_point));                  // dx

        cell_rhs(i) += (fe_values.shape_value(i, q_point) *  // phi_i(x_q)
                        rhs_values[q_point] *                // f(x_q)
                        fe_values.JxW(q_point));             // dx
      }

    cell->get_dof_indices(local_dof_indices);
    hanging_node_constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

    for (const auto i : local_dof_indices) diffusion_at_dof(i) = mu;
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  std::map<types::global_dof_index, double> boundary_values;
  Solution<dim> sol(m_settings.sol_id, m_settings.sol_freq);
  VectorTools::interpolate_boundary_values(
      dof_handler, {{0, &sol}, {1, &sol}, {2, &sol}}, boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                     system_rhs);
}

template <int dim>
void PoissonProblem<dim>::refine_grid(unsigned int cycle) {
  if (cycle == 0) {
    const auto mesh = itertools::split(m_settings.mesh, '.');
    if (mesh[0] == "Simplex") {
      const double sqrt3 = std::sqrt(3.0);
      const double sqrt6 = std::sqrt(6.0);
      GridGenerator::simplex(triangulation, {{-.5, -sqrt3 / 6., -sqrt6 / 12.},
                                             {.5, -sqrt3 / 6., -sqrt6 / 12.},
                                             {0, sqrt3 / 3., -sqrt6 / 12},
                                             {0, 0, sqrt6 / 4}});
    } else if (mesh[0] == "PlateWithHole") {
      GridGenerator::plate_with_a_hole(triangulation, .4, 1., 1., 1., 1., 1.,
                                       Point<dim>(), 0, 1, 1.,
                                       std::stoul(mesh[1]));
    } else if (mesh[0] == "HyperBall") {
      GridGenerator::hyper_ball(triangulation);
    } else if (mesh[0] == "HyperBallBalanced") {
      GridGenerator::hyper_ball_balanced(triangulation);
    } else if (mesh[0] == "Cylinder") {
      GridGenerator::subdivided_cylinder(triangulation, std::stoul(mesh[1]));
    } else if (mesh[0] == "Cube") {
      GridGenerator::hyper_cube(triangulation, -1.0, 1.0);
    } else if (mesh[0] == "Cheese") {
      GridGenerator::cheese(triangulation, {2, 2, 2});
    } else if (mesh[0] == "Torus") {
      GridGenerator::torus(triangulation, 2.0, 0.5);
    } else if (mesh[0] == "ReplicateHoles") {
      Triangulation<3> input;
      GridGenerator::hyper_cube_with_cylindrical_hole(input);
      GridGenerator::replicate_triangulation(input, {3, 2, 1}, triangulation);
    } else if (mesh[0] == "ReplicateCross") {
      Triangulation<3> input;
      GridGenerator::hyper_cross(input, {1, 1, 1, 2, 1, 2});
      GridGenerator::replicate_triangulation(input, {3, 2, 1}, triangulation);
    } else {
      std::cout << "Mesh name not recognised" << std::endl;
      std::exit(-1);
    }
    triangulation.refine_global(m_settings.num_bas_ref);
    const auto n = triangulation.n_cells();
    std::cout << "Basic tria has " << n << " cells." << std::endl;
    epsv.resize(n);

    std::uniform_real_distribution<double> distribution(
        0, m_settings.max_diffusion);
    std::default_random_engine generator(m_settings.seed);
    std::generate(epsv.begin(), epsv.end(),
                  [&]() { return distribution(generator); });

    std::cout << "Epsv description" << std::endl;
    math::describe(epsv);
  } else {
    triangulation.refine_global(1);
  }
}

template <int dim>
void PoissonProblem<dim>::output_solution(const unsigned int cycle) {
  const std::string vtk_filename =
      "solution-global-q" + std::to_string(fe.degree) + "-ref" +
      std::to_string(cycle) + "-mesh" + m_settings.mesh + "-nbr" +
      std::to_string(m_settings.num_bas_ref) + ".vtk";
  std::ofstream output(vtk_filename);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(diffusion_at_dof, "diffusion");
  data_out.build_patches();
  data_out.write_vtk(output);
}

template <int dim>
void PoissonProblem<dim>::print_convergence_table() {
  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_precision("Linfty", 3);

  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("Linfty", true);

  std::cout << std::endl;
  convergence_table.write_text(std::cout);

  {
    convergence_table.add_column_to_supercolumn("cycle", "n cells");
    convergence_table.add_column_to_supercolumn("cells", "n cells");

    std::vector<std::string> new_order;
    new_order.emplace_back("n cells");
    new_order.emplace_back("H1");
    new_order.emplace_back("L2");
    convergence_table.set_column_order(new_order);

    convergence_table.evaluate_convergence_rates(
        "L2", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
        "L2", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
        "H1", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
        "H1", ConvergenceTable::reduction_rate_log2);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);
  }
}

template <int dim>
void PoissonProblem<dim>::process_solution(const unsigned int cycle) {
  // error norms
  Vector<float> difference_per_cell(triangulation.n_active_cells());

  VectorTools::integrate_difference(
      dof_handler, solution,
      Solution<dim>(m_settings.sol_id, m_settings.sol_freq),
      difference_per_cell, QGauss<dim>(fe.degree + 1), VectorTools::L2_norm);
  const double L2_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::L2_norm);

  VectorTools::integrate_difference(
      dof_handler, solution,
      Solution<dim>(m_settings.sol_id, m_settings.sol_freq),
      difference_per_cell, QGauss<dim>(fe.degree + 1),
      VectorTools::H1_seminorm);
  const double H1_error = VectorTools::compute_global_error(
      triangulation, difference_per_cell, VectorTools::H1_seminorm);

  const unsigned int n_active_cells = triangulation.n_active_cells();
  const unsigned int n_dofs = dof_handler.n_dofs();

  std::cout << "Cycle " << cycle << ':' << std::endl
            << "   Number of active cells:       " << n_active_cells
            << std::endl
            << "   Number of degrees of freedom: " << n_dofs << std::endl;

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
}

template <int dim>
void PoissonProblem<dim>::output_stats(unsigned int cycle) {
  std::cout << " Active cells:   " << triangulation.n_active_cells()
            << std::endl;
  std::cout << " Number of dofs: " << dof_handler.n_dofs() << std::endl;
  std::time_t timestamp = std::time(nullptr);
  filestream << m_settings.settings_filename << "," << dim << ","
             << dof_handler.n_dofs() << "," << cycle << "," << fe.degree << ","
             << m_settings.sol_id << "," << m_settings.sol_freq << ","
             << m_settings.mesh << "," << m_settings.renumbering << ","
             << m_settings.seed << "," << m_settings.max_diffusion << ","
             << m_settings.num_bas_ref << "," << timestamp << ",";

  std::cout << "\nepsv: ";
  std::cout << "\ntimestamp: " << timestamp << std::endl;
}

template <int dim>
void PoissonProblem<dim>::run() {
  filestream.open(stats_filename, std::fstream::out | std::fstream::app);
  filestream << std::scientific << std::setprecision(17);
  filestream << "setting,dim,ndof,mesh_ref,degree,sol_id,freq,mesh,renumbering,"
                "seed,maxdiff,num_bas_ref,timestamp,";
  if (m_settings.make_view) {
    filestream << "t_view,view_size,view,view_count,view_max_pp,view_max_np\n";
  } else {
    filestream << "theta,maxrowsum,symop,naggr,tol,t_amg_setup,";
    if (m_settings.output_setup_details) {
      filestream << "nrows,nze,sparsity,grid,operator,memory,";
    }
    filestream << "t_solve,niters,p_res\n";
  }

  const unsigned int n_cycles = m_settings.cycles;
  for (unsigned int cycle = 0; cycle < n_cycles; ++cycle) {
    refine_grid(cycle);
    setup_system();
    assemble_system();

    if (m_settings.make_view) {
      output_stats(cycle);
      std::cout << "Making view" << std::endl;
      m_viewmaker.make_view(filestream, system_matrix);
      m_viewmaker.print_view(filestream);
    } else {
      const auto& tv = m_settings.strong_threshold;
      for (double t = tv[0]; t <= tv[1]; t += tv[2]) {
        output_stats(cycle);
        solution = zero_solution;  // set first iteration x^{0} = 0.0
        const BoomerAMGData data(true, t, 0.9, 0,
                                 m_settings.output_setup_details);
        std::cout << "Solving" << std::endl;
        const auto it = amg_solver::amg_solve(
            data, m_settings.toll, filestream, system_matrix, system_rhs,
            solution, mpi_communicator, stats_filename,
            hanging_node_constraints);
        std::cout << "Converged in " << it << " iterations" << std::endl;
      }

      if (m_settings.evaluate_errors) process_solution(cycle);
      if (m_settings.output_results) output_solution(cycle);
    }
  }

  if (m_settings.evaluate_errors) print_convergence_table();
}
}  // namespace MyStructuredDiffusion

int main(int argc, char* argv[]) {
  try {
    using namespace dealii;
    using namespace MyStructuredDiffusion;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const auto settings =
        get_settings(argc > 1 ? std::string(argv[1]) : "../settings.json");

    if (settings.dim == 2) {
      // PoissonProblem<2> problem(settings);
      // problem.run();
    } else if (settings.dim == 3) {
      PoissonProblem<3> problem(settings);
      problem.run();
    } else {
      std::cout << "Unhandled dimesion: " << settings.dim << std::endl;
      std::exit(-1);
    }
    std::cout << "end!" << std::endl;
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
  }
  return 0;
}