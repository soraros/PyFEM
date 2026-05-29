############################################################################
#  Skim: ch.3 cantilever8 — FiniteStrainContinuum + NonlinearSolver
############################################################################

input = "../../examples/ch03/cantilever8.dat";

ContElem =
{
  type = "FiniteStrainContinuum";

  material =
  {
    type = "PlaneStress";
    E    = 100.0;
    nu   = 0.3;
  };
};

solver =
{
  type = "NonlinearSolver";

  fixedStep = true;
  maxCycle   = 20;
};

outputModules = ["output"];

output =
{
  type = "OutputWriter";
};
