############################################################################
#  Skim: ch.4 shallow truss — Truss + Spring + RiksSolver
############################################################################

input = "../../examples/ch04/ShallowtrussRiks.dat";

TrussElem  =
{
  type = "Truss";
  E    = 5e6;
  Area = 1.0;
};

SpringElem =
{
  type = "Spring";
  k    = 100.0;
};

solver =
{
  type = "RiksSolver";

  fixedStep = true;
  maxLam    = 10.0;
};

outputModules = ["output"];

output =
{
  type = "OutputWriter";
};
