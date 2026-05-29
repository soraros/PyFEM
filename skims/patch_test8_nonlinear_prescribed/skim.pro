############################################################################
#  Skim: PatchTest8 prescribed field + displacement-driven ramp
############################################################################

input = "../../examples/ch02/PatchTest8.dat";

ContElem =
{
  type = "SmallStrainContinuum";

  material =
  {
    type = "PlaneStress";
    E    = 1.e6;
    nu   = 0.25;
  };
};

solver =
{
  type = "NonlinearSolver";

  tol = 1.0e-10;
  iterMax = 10;
  loadTable = [0.25, 0.5, 0.75, 1.0];
};

outputModules = ["output"];

output =
{
  type = "OutputWriter";
};
