############################################################################
#  Skim input for v3 parity — same physics as examples/ch02/PatchTest8.pro
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
  type = "LinearSolver";
};

outputModules = ["vtk" , "output" ];

vtk =
{
  type = "MeshWriter";
};

output =
{
  type = "OutputWriter";
};
