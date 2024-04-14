#include "settings.h"

#include <deal.II/base/parameter_handler.h>

#include "../../common/myutils.h"

Settings get_settings(const std::string& filename) {
  using namespace dealii;
  ParameterHandler prm;
  prm.declare_entry("dim", "3", Patterns::Integer());
  prm.declare_entry("deg", "1", Patterns::Integer());
  prm.declare_entry("num refinements", "3", Patterns::Integer());
  prm.declare_entry("cycles", "1", Patterns::Integer());
  prm.declare_entry("renumbering", "0", Patterns::Integer());

  prm.declare_entry("seed", "0", Patterns::Integer());
  prm.declare_entry("mode", "1", Patterns::Integer());
  prm.declare_entry("pattern size", "2", Patterns::Integer());
  prm.declare_entry("max young exp", "6", Patterns::Double());
  prm.declare_entry("sharp", "false", Patterns::Bool());

  prm.declare_entry("tol", "1e-8", Patterns::Double());
  prm.declare_entry("strong threshold", "");

  prm.declare_entry("make view", "false", Patterns::Bool());
  prm.declare_entry("view size", "50", Patterns::Integer());

  prm.declare_entry("evaluate errors", "true", Patterns::Bool());
  prm.declare_entry("output results", "true", Patterns::Bool());
  prm.declare_entry("stats filename", "stats.csv", Patterns::FileName());

  prm.declare_entry("output setup details", "true", Patterns::Bool());

  prm.parse_input(filename);

  const auto thetav = itertools::map_to_array<3, std::string, double>(
      itertools::split(prm.get("strong threshold"), ','),
      [](const auto& x) { return std::stod(x); });

  return Settings{
      prm.get_integer("dim"),
      prm.get_integer("deg"),
      prm.get_integer("num refinements"),
      prm.get_integer("cycles"),
      prm.get_integer("renumbering"),

      prm.get_integer("seed"),
      prm.get_integer("mode"),
      prm.get_integer("pattern size"),
      prm.get_double("max young exp"),
      prm.get_bool("sharp"),

      prm.get_double("tol"),
      thetav,

      prm.get_bool("make view"),
      prm.get_integer("view size"),

      prm.get_bool("evaluate errors"),
      prm.get_bool("output results"),
      prm.get("stats filename"),
      filename,

      prm.get_bool("output setup details"),
  };
}
