"""Internal backend-neutral preparation and reference assembly surface."""

from pyfem.v3.assembly.contracts import (
  AssemblyPlanProvenance,
  CanonicalCooOperator,
  DomainCooPlan,
  LinearStaticContributionRequest,
  LinearStaticContributions,
  NodalVectorContributionPlan,
  PreparedAssemblyPlan,
)
from pyfem.v3.assembly.diagnostics import (
  AssemblyEvaluationDiagnostic,
  AssemblyEvaluationError,
  AssemblyPreparationDiagnostic,
  AssemblyPreparationError,
)
from pyfem.v3.assembly.prepare import (
  ASSEMBLY_GEOMETRY_POLICY,
  ASSEMBLY_REDUCTION_POLICY,
  PREPARED_ASSEMBLY_PLAN_MANIFEST_SCHEMA,
  prepare_assembly_plan,
)
from pyfem.v3.assembly.reference import assemble_reference_linear

__all__ = [
  "ASSEMBLY_GEOMETRY_POLICY",
  "ASSEMBLY_REDUCTION_POLICY",
  "PREPARED_ASSEMBLY_PLAN_MANIFEST_SCHEMA",
  "AssemblyEvaluationDiagnostic",
  "AssemblyEvaluationError",
  "AssemblyPlanProvenance",
  "AssemblyPreparationDiagnostic",
  "AssemblyPreparationError",
  "CanonicalCooOperator",
  "DomainCooPlan",
  "LinearStaticContributionRequest",
  "LinearStaticContributions",
  "NodalVectorContributionPlan",
  "PreparedAssemblyPlan",
  "assemble_reference_linear",
  "prepare_assembly_plan",
]
