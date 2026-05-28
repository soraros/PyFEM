############################################################################
#  Skim: PatchTest8 with nodal load (parity vs legacy LinearSolver)
############################################################################

input = "PatchTest8_loaded.dat";

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
  type = "LinearSolver";
};

outputModules = ["vtk", "output"];

vtk =
{
  type = "MeshWriter";
};

output =
{
  type = "OutputWriter";
};
