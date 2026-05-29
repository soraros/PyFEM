############################################################################
#  Skim input for v3 parity — PatchTest8 mesh with PlaneStrain material
############################################################################

input = "../../examples/ch02/PatchTest8.dat";

ContElem =
{
  type = "SmallStrainContinuum";

  material =
  {
    type = "PlaneStrain";
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
