############################################################################
#  Skim: PatchTest8 loaded + NonlinearSolver (single-step load table)
############################################################################

input = "../patch_test8_loaded/PatchTest8_loaded.dat";

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
  loadTable = [1.0];
};

outputModules = ["output"];

output =
{
  type = "OutputWriter";
};
