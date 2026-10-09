############################################################################
#  Skim: contact_test02 — penalty contact strip (trimmed for v3 parity)
#
#  Trimmed from examples/contact/contact_test02.pro: the output writers are
#  dropped to the no-op set, the solver block pins tol/iterMax and a
#  loadTable replicating the original dtime=0.2 ramp's first five cycles,
#  and the element block reads SmallStrainContinuum — the v3 finite-strain
#  slice is serendipity-quad8-only and rejects this quad4 mesh, so both
#  sides run the trimmed deck and parity is exact by construction. The c1
#  Contact block remains the legacy oracle input; the v3 side wires the
#  same obstacle through pyfem.v3.compile.contact (converter support is
#  deliberately out of scope), reading only the .dat and the solver
#  settings.
############################################################################

input = "contact_test02.dat";

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

models = ["c1"];

c1 =
{
  type = "Contact";

  object    = "disc";

  radius    = 1.0;
  centre    = [ 8.0 , 1.4 ];
  direction = [0.,-0.1];
  penalty   = 1.0e6;
};

solver =
{
  type     = "NonlinearSolver";
  tol      = 1.0e-8;
  iterMax  = 25;
  loadTable = [0.2, 0.4, 0.6, 0.8, 1.0];
};

outputModules = ["output"];

output =
{
  type = "OutputWriter";
};
