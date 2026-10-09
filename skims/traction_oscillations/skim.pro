############################################################################
#  Skim: TractionOscillations — trimmed linear cohesive-transfer oracle.
#
#  The examples/ch13/TractionOscillations.pro physics (PlaneStress strips,
#  linear Dummy interface, LinearSolver) trimmed to 6 segments (24 quad4 + 6
#  interface elements across the whole seam). Two symmetric top-corner loads
#  transfer through the interface into the minimally constrained substrate —
#  the configuration whose Gauss-point tractions oscillate along the seam in
#  the book's section 13.2 discussion.
############################################################################

input = "traction_oscillations.dat";

ContElem =
{
  type = "SmallStrainContinuum";

  material =
  {
    type = "PlaneStress";
    E    = 1.0e6;
    nu   = 0.25;
  };
};

InterfaceElem =
{
  type = "Interface";

  material =
  {
    type = "Dummy";
    D    = 1.0e5;
  };
};

solver =
{
  type = "LinearSolver";
};

outputModules = ["output"];

output =
{
  type = "OutputWriter";
};
