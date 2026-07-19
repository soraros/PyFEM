"""Immutable public solution for the frozen Q8 linear-static flow."""

from __future__ import annotations

from dataclasses import dataclass

from pyfem.v3.analysis.contracts import (
  AcceptedTransition,
  LinearConvergenceRecord,
  LinearStatic,
)
from pyfem.v3.model import (
  CanonicalManifest,
  CommittedAnalysisState,
  CompiledModel,
  CompiledProgram,
  ContentFingerprint,
  FinalizedArray,
  InstanceId,
)
from pyfem.v3.results.contracts import LinearBalanceLedger, VerificationReport
from pyfem.v3.results.diagnostics import (
  SolutionVerificationDiagnostic,
  SolutionVerificationError,
)
from pyfem.v3.results.verification import fresh_verify, verify_record_data

_RAW_RESULT_ERRORS = (
  AttributeError,
  IndexError,
  KeyError,
  OverflowError,
  RecursionError,
  TypeError,
  ValueError,
)


def _unexpected_result_failure() -> SolutionVerificationError:
  return SolutionVerificationError(
    (
      SolutionVerificationDiagnostic(
        "malformed-solution-record",
        "solution is partially initialized or structurally malformed",
      ),
    )
  )


@dataclass(frozen=True, slots=True, eq=False)
class Solution:
  """One detached accepted linear-static result with fresh verification."""

  model: CompiledModel
  program: CompiledProgram
  request: LinearStatic
  request_manifest: CanonicalManifest
  prepared_instance_id: InstanceId
  plan_content_fingerprint: ContentFingerprint
  state: CommittedAnalysisState
  transition: AcceptedTransition
  convergence: LinearConvergenceRecord
  ledger: LinearBalanceLedger

  @property
  def primary_values(self) -> FinalizedArray:
    """Return the immutable full primary-vector carrier."""
    return self.state.physical.primary_values

  def verify_record(self) -> VerificationReport:
    """Check retained structure and algebra without recomputing equilibrium."""
    try:
      return verify_record_data(
        model=self.model,
        program=self.program,
        request=self.request,
        request_manifest=self.request_manifest,
        prepared_instance_id=self.prepared_instance_id,
        plan_content_fingerprint=self.plan_content_fingerprint,
        state=self.state,
        transition=self.transition,
        convergence=self.convergence,
        ledger=self.ledger,
      )
    except SolutionVerificationError:
      raise
    except _RAW_RESULT_ERRORS as error:
      raise _unexpected_result_failure() from error

  def verify(self) -> VerificationReport:
    """Rebuild the P1-B plan and response in a fresh verification workspace."""
    record = self.verify_record()
    if not record.passed:
      return record
    try:
      return fresh_verify(
        model=self.model,
        program=self.program,
        request_manifest=self.request_manifest,
        prepared_instance_id=self.prepared_instance_id,
        plan_content_fingerprint=self.plan_content_fingerprint,
        transaction_id=self.transition.transaction_id,
        trial_id=self.transition.trial_id,
        base_generation=self.transition.base_generation,
        candidate_generation=self.transition.candidate_generation,
        state=self.state,
        retained_ledger=self.ledger,
        convergence=self.convergence,
      )
    except SolutionVerificationError:
      raise
    except _RAW_RESULT_ERRORS as error:
      raise _unexpected_result_failure() from error
