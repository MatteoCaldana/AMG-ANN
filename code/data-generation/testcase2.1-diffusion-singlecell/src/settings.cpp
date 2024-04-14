#include "settings.h"

#include <deal.II/base/parameter_handler.h>

#include "../../common/myutils.h"

Settings get_settings(const std::string& filename) {
  using namespace dealii;
  ParameterHandler prm;
  prm.declare_entry("dim", "2", Patterns::Integer());
  prm.declare_entry("degree", "1", Patterns::Integer());

  prm.declare_entry("dof renumbering", "0", Patterns::Integer());
  prm.declare_entry("diffusion exp", "10", Patterns::Double());
  prm.declare_entry("num ref", "1", Patterns::Integer());
  prm.declare_entry("marked point", "0,0,0");
  prm.declare_entry("toll", "1e-12", Patterns::Double());
  prm.declare_entry("eliminate cols", "true", Patterns::Bool());
  prm.declare_entry("hermitian", "true", Patterns::Bool());

  prm.declare_entry("output results", "true", Patterns::Bool());
  prm.declare_entry("view size", "0", Patterns::Integer());
  prm.declare_entry("solver mode", "0", Patterns::Integer());
  prm.declare_entry("stats filename", "stats.csv", Patterns::FileName());
  prm.parse_input(filename);

  const auto mpt = itertools::map_to_array<3, std::string, double>(
      itertools::split(prm.get("marked point"), ','),
      [](const auto& x) { return std::stod(x); });

  prm.declare_entry("output setup details", "true", Patterns::Bool());

  return Settings{
      prm.get_integer("dim"),
      prm.get_integer("degree"),

      prm.get_integer("dof renumbering"),
      prm.get_double("diffusion exp"),
      prm.get_integer("num ref"),
      mpt,
      prm.get_double("toll"),
      prm.get_bool("hermitian"),

      prm.get_bool("output results"),
      prm.get_integer("view size"),
      prm.get_integer("solver mode"),
      prm.get("stats filename"),
      filename,
      prm.get_bool("output setup details"),
  };
}
