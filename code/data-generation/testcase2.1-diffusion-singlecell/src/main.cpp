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
#include <deal.II/lac/slepc_solver.h>
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
#include "../../common/view_maker.h"
#include "settings.h"

template <int dim>
class RightHandSide : public Function<dim> {
 public:
  virtual double value(const Point<dim>& p,
                       const unsigned int) const override {
    constexpr double f = 2.0;
    double ss = 1.0;
    for (int i = 0; i < dim; ++i) ss *= std::sin(f * M_PI * p[i]);
    return 2 * f * f * M_PI * M_PI * ss;
  }
};

namespace SingleCellDiffusion {
using namespace dealii;

template <int dim>
class PoissonProblem {
 public:
  PoissonProblem(const Settings& s);

  void run();

 private:
  void setup_system();
  void assemble_system();
  void refine_grid();
  void output_solution();
  void print_convergence_table();
  void output_stats();

  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;

  Triangulation<dim> triangulation;
  DoFHandler<dim> dof_handler;
  FE_Q<dim> fe;

  AffineConstraints<double> hanging_node_constraints;

  PETScWrappers::MPI::SparseMatrix system_matrix;

  PETScWrappers::MPI::Vector solution, m_zero_solution;
  PETScWrappers::MPI::Vector system_rhs;
  Vector<double> diffusion_at_dof;

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
      break;
    case 1:
      DoFRenumbering::Cuthill_McKee(dof_handler);
      break;
    case 2:
      DoFRenumbering::boost::king_ordering(dof_handler);
      break;
    default:
      std::cout << "WARNING: no valid reordering is specified! Doing nothing."
                << std::endl;
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
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
  diffusion_at_dof.reinit(dof_handler.n_dofs());
  diffusion_at_dof = -1.0;
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

  const auto hi = std::pow(10.0, m_settings.diffusion);

  const RightHandSide<dim> right_hand_side;
  std::vector<double> rhs_values(n_q_points);

  const Point<dim> marked_point =
      dim == 2
          ? Point<dim>(m_settings.marked_point[0], m_settings.marked_point[1])
          : Point<dim>(m_settings.marked_point[0], m_settings.marked_point[1],
                       m_settings.marked_point[2]);
  for (const auto& cell : dof_handler.active_cell_iterators()) {
    cell_matrix = 0.;
    cell_rhs = 0.;

    fe_values.reinit(cell);
    const auto marked_cell = cell->point_inside(marked_point);
    const auto mu = marked_cell ? hi : 1.0;

    right_hand_side.value_list(fe_values.get_quadrature_points(), rhs_values);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          cell_matrix(i, j) +=
              ((mu * fe_values.shape_grad(i, q_point) *  // grad phi_i(x_q)
                fe_values.shape_grad(j, q_point)) *      // grad phi_j(x_q)
               fe_values.JxW(q_point));                  // dx

        cell_rhs(i) += mu * fe_values.shape_value(i, q_point) *  // phi_i(x_q)
                       rhs_values[q_point] *                     // f(x_q)
                       fe_values.JxW(q_point);                   // dx
      }

    cell->get_dof_indices(local_dof_indices);
    hanging_node_constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);

    for (const auto i : local_dof_indices) {
      if (diffusion_at_dof(i) < 0 || mu == hi) {
        diffusion_at_dof(i) = mu;
      }
    }
  }

  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                     system_rhs, false);

  m_zero_solution = solution;
}

template <int dim>
void PoissonProblem<dim>::refine_grid() {
  GridGenerator::hyper_cube(triangulation, -1., 1.);
  triangulation.refine_global(m_settings.num_ref);
  const auto n = triangulation.n_cells();
  std::cout << "Basic tria has " << n << " cells." << std::endl;
}

template <int dim>
void PoissonProblem<dim>::output_solution() {
  const std::string vtk_filename =
      "solution-global-q" + std::to_string(fe.degree) + "-ref" +
      std::to_string(m_settings.num_ref) + "-pt" +
      std::to_string(m_settings.marked_point[0]) + "-" +
      std::to_string(m_settings.marked_point[1]) + "-" +
      std::to_string(m_settings.marked_point[2]) + +".vtk";
  std::ofstream output(vtk_filename);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.add_data_vector(diffusion_at_dof, "diffusion");
  data_out.build_patches();
  data_out.write_vtk(output);
}

template <int dim>
void PoissonProblem<dim>::output_stats() {
  std::cout << " Active cells:   " << triangulation.n_active_cells()
            << std::endl;
  std::cout << " Number of dofs: " << dof_handler.n_dofs() << std::endl;
  std::time_t timestamp = std::time(nullptr);
  filestream << m_settings.settings_filename << "," << dim << ","
             << dof_handler.n_dofs() << "," << fe.degree << ","
             << m_settings.renumbering << "," << m_settings.diffusion << ","
             << timestamp << ",";

  std::cout << "\nepsv: ";
  std::cout << "\ntimestamp: " << timestamp << std::endl;
}

template <int dim>
void PoissonProblem<dim>::run() {
  filestream.open(stats_filename, std::fstream::out | std::fstream::app);
  filestream << std::scientific << std::setprecision(17);

  refine_grid();
  setup_system();
  assemble_system();

  if (m_settings.solver_mode == 0) {
    filestream << "timestamp,setting,dim,ndof,mesh_ref,degree,renumbering,"
                  "diff,mpt,toll,solver,min,max,amin,amax\n";

    std::cout << "Start eigensolvers" << std::endl;
    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
    std::vector<double> eigenvalues;

    std::cout << "Memory allocaltion" << std::endl;
    eigenfunctions.resize(1);
    const IndexSet eigenfunction_index_set = dof_handler.locally_owned_dofs();
    for (unsigned int i = 0; i < eigenfunctions.size(); ++i)
      eigenfunctions[i].reinit(eigenfunction_index_set, MPI_COMM_WORLD);

    eigenvalues.resize(eigenfunctions.size());

    SolverControl solver_control(dof_handler.n_dofs(), m_settings.toll);
    std::vector<std::shared_ptr<SLEPcWrappers::SolverBase>> solvers = {
        std::make_shared<SLEPcWrappers::SolverArnoldi>(solver_control),
        std::make_shared<SLEPcWrappers::SolverKrylovSchur>(solver_control),
        std::make_shared<SLEPcWrappers::SolverLAPACK>(solver_control),
    };
    for (size_t i = 0; i < solvers.size(); ++i) {
      std::cout << std::time(nullptr) << " Solver #" << i << std::endl;
      auto eigensolver_ptr = solvers[i];
      eigensolver_ptr->set_problem_type(EPS_NHEP);

      std::time_t timestamp = std::time(nullptr);
      filestream << timestamp << ",";
      filestream << m_settings.settings_filename << ",";
      filestream << dim << ",";
      filestream << dof_handler.n_dofs() << ",";
      filestream << m_settings.num_ref << ",";
      filestream << fe.degree << ",";
      filestream << m_settings.renumbering << ",";
      filestream << m_settings.diffusion << ",";
      itertools::print(m_settings.marked_point, filestream);
      filestream << ",";
      filestream << m_settings.toll << ",";
      filestream << i << ",";
      for (auto mode : {EPS_SMALLEST_REAL, EPS_LARGEST_REAL,
                        EPS_SMALLEST_MAGNITUDE, EPS_SMALLEST_MAGNITUDE}) {
        eigensolver_ptr->set_which_eigenpairs(mode);
        eigensolver_ptr->solve(system_matrix, eigenvalues, eigenfunctions,
                               eigenfunctions.size());
        filestream << eigenvalues[0] << ",";
      }
      std::cout << std::time(nullptr) << " Done" << std::endl;
      filestream << "\n" << std::flush;
    }

  } else if (m_settings.solver_mode == 1) {
    filestream << "timestamp,setting,dim,ndof,mesh_ref,degree,renumbering,"
                  "diff,mpt,theta,mrs,sym,agg_lvls,tol,t_setup,nrows,nze,spa,"
                  "grid,op,mem,t_solve,it,res\n";

    for (double t = 0.05; t <= 0.95; t += 0.0125) {
      solution = m_zero_solution;  // set first iteration x^{0} = 0.0
      std::time_t timestamp = std::time(nullptr);
      filestream << timestamp << ",";
      filestream << m_settings.settings_filename << ",";
      filestream << dim << ",";
      filestream << dof_handler.n_dofs() << ",";
      filestream << m_settings.num_ref << ",";
      filestream << fe.degree << ",";
      filestream << m_settings.renumbering << ",";
      filestream << m_settings.diffusion << ",";
      itertools::print(m_settings.marked_point, filestream);
      filestream << ",";
      using BoomerAMGData =
          PETScWrappers::PreconditionBoomerAMG::AdditionalData;

      const BoomerAMGData data(true,  //  symmetric_operator=false,
                               t,     //  strong_threshold=0.25
                               0.9,   //  max_row_sum=0.9
                               0,     //  aggressive_coarsening_num_levels=0
                               true   //  output_details=false
      );
      amg_solver::amg_solve(data, 1e-8, filestream, system_matrix, system_rhs,
                            solution, mpi_communicator, stats_filename + ".tmp",
                            hanging_node_constraints);
    }
  } else if (m_settings.solver_mode == 2) {
    filestream << "timestamp,setting,dim,ndof,mesh_ref,degree,renumbering,"
                  "diff,mpt,t_view,view_size,view,view_count,view_max_pp,view_"
                  "max_np\n";
    solution = m_zero_solution;  // set first iteration x^{0} = 0.0
    std::time_t timestamp = std::time(nullptr);
    filestream << timestamp << ",";
    filestream << m_settings.settings_filename << ",";
    filestream << dim << ",";
    filestream << dof_handler.n_dofs() << ",";
    filestream << m_settings.num_ref << ",";
    filestream << fe.degree << ",";
    filestream << m_settings.renumbering << ",";
    filestream << m_settings.diffusion << ",";
    itertools::print(m_settings.marked_point, filestream);
    filestream << ",";

    m_viewmaker.make_view(filestream, system_matrix);
    m_viewmaker.print_view(filestream);

  } else {
    std::cout << "Unknown solver mode: " << m_settings.solver_mode << std::endl;
    std::exit(-1);
  }

  if (m_settings.output_results) output_solution();
}
}  // namespace SingleCellDiffusion

int main(int argc, char* argv[]) {
  try {
    using namespace dealii;
    using namespace SingleCellDiffusion;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    const auto settings =
        get_settings(argc > 1 ? std::string(argv[1]) : "../settings.json");

    if (settings.dim == 2) {
      PoissonProblem<2> problem(settings);
      problem.run();
    } else if (settings.dim == 3) {
      PoissonProblem<3> problem(settings);
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
  }
  return 0;
}