############################################################################
#  Skim input for v3 parity — same physics as PatchTest8_3D.pro
############################################################################

input = "../../examples/ch02/PatchTest8_3D.dat";

ContElem =
{
  type = "SmallStrainContinuum";

  material =
  {
    type = "Isotropic";
    E    = 1.e6;
    nu   = 0.25;
  };
};

solver =
{
  type = "LinearSolver";
};

outputModules = ["vtk","output"];

vtk =
{
  type = "MeshWriter";
};

output =
{
  type = "OutputWriter";

  onScreen = true;
};
