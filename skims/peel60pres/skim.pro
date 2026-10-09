############################################################################
#  Skim: PeelTest60pres — trimmed prescribed-displacement cohesive peel.
#
#  The examples/ch13/PeelTest60pres.pro physics (PlaneStrain strips,
#  XuNeedleman interface, NonlinearSolver, prescribed end displacements)
#  trimmed to 6 segments (24 quad4 + 4 interface elements, the last two
#  segments pre-cracked) with the ramp resolved into a loadTable. The ramp
#  peaks past the Xu-Needleman strength (vnmax = Gc/(e*Tult) = 0.0736 at the
#  deck's Gc = 0.1, Tult = 0.5): lam = 0.75 sits just below the peak opening
#  0.0736 (v = 0.0736 at lam = 0.736) and lam = 2.0 reaches 2.7x vnmax, deep
#  into softening. The legacy solver block's `lam = 100.0` key is a no-op on
#  the legacy side (NonlinearSolver.py never reads it) and is dropped here.
############################################################################

input = "peel60pres.dat";

ContElem =
{
  type = "SmallStrainContinuum";

  material =
  {
    type = "PlaneStrain";
    E    = 100.0;
    nu   = 0.3;
  };
};

InterfaceElem =
{
  type = "Interface";

  material =
  {
    type = "XuNeedleman";

    Tult = 0.5;
    Gc   = 0.1;
  };
};

solver =
{
  type = "NonlinearSolver";

  tol       = 1.0e-10;
  iterMax   = 25;
  loadTable = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0];
};

outputModules = ["output"];

output =
{
  type = "OutputWriter";
};
