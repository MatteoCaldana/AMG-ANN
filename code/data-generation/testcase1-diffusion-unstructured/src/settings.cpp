#include "settings.h"

#include <deal.II/base/parameter_handler.h>

#include "../../common/myutils.h"

Settings get_settings(const std::string& filename) {
  using namespace dealii;
  ParameterHandler prm;
  prm.declare_entry("dim", "2", Patterns::Integer());
  prm.declare_entry("degree", "1", Patterns::Integer());

  prm.declare_entry("mesh filename", "mesh.msh", Patterns::FileName());
  prm.declare_entry("dof renumbering", "0", Patterns::Integer());
  prm.declare_entry("solution id", "0", Patterns::Integer());
  prm.declare_entry("solution freq", "1", Patterns::Double());
  prm.declare_entry("random seed", "0", Patterns::Integer());
  prm.declare_entry("max diffusion exp", "10", Patterns::Double());
  prm.declare_entry("ncycles", "1", Patterns::Integer());
  prm.declare_entry("num base ref", "1", Patterns::Integer());

  prm.declare_entry("strong threshold", "");
  prm.declare_entry("toll", "1e-8", Patterns::Double());

  prm.declare_entry("output results", "true", Patterns::Bool());
  prm.declare_entry("evaluate errors", "true", Patterns::Bool());

  prm.declare_entry("make view", "true", Patterns::Bool());
  prm.declare_entry("view size", "50", Patterns::Integer());

  prm.declare_entry("stats filename", "stats.csv", Patterns::FileName());

  prm.declare_entry("output setup details", "true", Patterns::Bool());

  prm.parse_input(filename);

  const auto thetav = itertools::map_to_array<3, std::string, double>(
      itertools::split(prm.get("strong threshold"), ','),
      [](const auto& x) { return std::stod(x); });

  return Settings{
      prm.get_integer("dim"),
      prm.get_integer("degree"),
      prm.get("mesh filename"),
      prm.get_integer("dof renumbering"),
      prm.get_integer("solution id"),
      prm.get_double("solution freq"),
      prm.get_integer("random seed"),
      prm.get_double("max diffusion exp"),
      prm.get_integer("num base ref"),
      prm.get_integer("ncycles"),
      thetav,
      prm.get_double("toll"),
      prm.get_bool("output results"),
      prm.get_bool("evaluate errors"),
      prm.get_bool("make view"),
      prm.get_integer("view size"),
      prm.get("stats filename"),
      filename,
      prm.get_bool("output setup details"),
  };
}
