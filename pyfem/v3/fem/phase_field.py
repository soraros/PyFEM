"""Batched plane-strain AT2 spectral response and scalar phase oracles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class PhaseFieldPreparationError(ValueError):
  """Raised when a branch-consistent phase-field tangent is not defined."""

  def __init__(self, code: str, message: str) -> None:
    super().__init__(message)
    self.code = code


@dataclass(frozen=True, slots=True)
class SpectralAT2Response:
  """Energies, physical Voigt stresses, and engineering-strain tangents."""

  positive_energy: np.ndarray
  negative_energy: np.ndarray
  positive_stress: np.ndarray
  negative_stress: np.ndarray
  positive_tangent: np.ndarray | None
  negative_tangent: np.ndarray | None
  eigenvalues: np.ndarray


_STRAIN_BASIS = np.array(
  [
    [[1.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 1.0]],
    [[0.0, 0.5], [0.5, 0.0]],
  ],
  dtype=np.float64,
)
_TRACE_BASIS = np.array([1.0, 1.0, 0.0], dtype=np.float64)


def _macaulay(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  absolute = np.abs(values)
  return 0.5 * (values + absolute), 0.5 * (values - absolute)


def _stress_voigt(tensor: np.ndarray) -> np.ndarray:
  return np.stack((tensor[:, 0, 0], tensor[:, 1, 1], tensor[:, 0, 1]), axis=1)


def _spectral_tangents(
  eigenvalues: np.ndarray,
  eigenvectors: np.ndarray,
  trace: np.ndarray,
  lame_lambda: float,
  shear_modulus: float,
) -> tuple[np.ndarray, np.ndarray]:
  positive_eigenvalues, negative_eigenvalues = _macaulay(eigenvalues)
  positive_trace, negative_trace = _macaulay(trace)
  local_basis = np.einsum(
    "nji,ajk,nkl->nail",
    eigenvectors,
    _STRAIN_BASIS,
    eigenvectors,
    optimize=True,
  )
  local_positive = np.zeros((eigenvalues.shape[0], 3, 2, 2), dtype=np.float64)
  local_negative = np.zeros_like(local_positive)
  for index in range(2):
    positive_active = (eigenvalues[:, index] > 0.0).astype(np.float64)
    negative_active = (eigenvalues[:, index] < 0.0).astype(np.float64)
    local_positive[:, :, index, index] = (
      lame_lambda * (positive_trace > 0.0)[:, None] * _TRACE_BASIS[None, :]
      + 2.0 * shear_modulus * positive_active[:, None] * local_basis[:, :, index, index]
    )
    local_negative[:, :, index, index] = (
      lame_lambda * (negative_trace < 0.0)[:, None] * _TRACE_BASIS[None, :]
      + 2.0 * shear_modulus * negative_active[:, None] * local_basis[:, :, index, index]
    )

  difference = eigenvalues[:, 1] - eigenvalues[:, 0]
  repeated = difference == 0.0
  positive_ratio = np.empty_like(difference)
  negative_ratio = np.empty_like(difference)
  np.divide(
    2.0 * shear_modulus * (positive_eigenvalues[:, 1] - positive_eigenvalues[:, 0]),
    difference,
    out=positive_ratio,
    where=~repeated,
  )
  np.divide(
    2.0 * shear_modulus * (negative_eigenvalues[:, 1] - negative_eigenvalues[:, 0]),
    difference,
    out=negative_ratio,
    where=~repeated,
  )
  positive_ratio[repeated] = 2.0 * shear_modulus * (eigenvalues[repeated, 0] > 0.0)
  negative_ratio[repeated] = 2.0 * shear_modulus * (eigenvalues[repeated, 0] < 0.0)
  for local, ratio in (
    (local_positive, positive_ratio),
    (local_negative, negative_ratio),
  ):
    local[:, :, 0, 1] = ratio[:, None] * local_basis[:, :, 0, 1]
    local[:, :, 1, 0] = local[:, :, 0, 1]

  tangents = []
  for local in (local_positive, local_negative):
    tensor = np.einsum(
      "nip,napq,njq->naij",
      eigenvectors,
      local,
      eigenvectors,
      optimize=True,
    )
    tangents.append(
      _stress_voigt(tensor.reshape(-1, 2, 2)).reshape(-1, 3, 3).transpose(0, 2, 1)
    )
  return tangents[0], tangents[1]


def spectral_at2_response(
  strains: np.ndarray,
  lame_lambda: float,
  shear_modulus: float,
  *,
  tangent: bool = True,
) -> SpectralAT2Response:
  """Evaluate the two-dimensional spectral split over ``(entity, 3)`` strains.

  The Voigt order is ``xx, yy, xy`` with engineering input shear and physical
  output shear. Positive and negative Macaulay parts are exactly
  ``(value +/- abs(value))/2``. A tangent request at a zero eigenvalue or zero
  trace fails explicitly because the branch derivative is not unique there.
  """
  if (
    type(strains) is not np.ndarray
    or strains.dtype != np.dtype(np.float64)
    or strains.dtype.metadata is not None
    or strains.ndim != 2
    or strains.shape[1] != 3
  ):
    msg = "AT2 spectral strains must be a plain (entity, 3) float64 array"
    raise TypeError(msg)
  if not bool(np.isfinite(strains).all()):
    msg = "AT2 spectral strains must be finite"
    raise ValueError(msg)
  if not np.isfinite(lame_lambda) or not np.isfinite(shear_modulus):
    msg = "AT2 elastic moduli must be finite"
    raise ValueError(msg)
  if lame_lambda + 2.0 * shear_modulus <= 0.0 or shear_modulus <= 0.0:
    msg = "AT2 elastic moduli must define positive plane-strain elasticity"
    raise ValueError(msg)

  tensor = np.empty((strains.shape[0], 2, 2), dtype=np.float64)
  tensor[:, 0, 0] = strains[:, 0]
  tensor[:, 1, 1] = strains[:, 1]
  tensor[:, 0, 1] = 0.5 * strains[:, 2]
  tensor[:, 1, 0] = tensor[:, 0, 1]
  eigenvalues, eigenvectors = np.linalg.eigh(tensor)
  trace = strains[:, 0] + strains[:, 1]
  if tangent and (bool(np.any(eigenvalues == 0.0)) or bool(np.any(trace == 0.0))):
    raise PhaseFieldPreparationError(
      "phase-field-spectral-kink",
      "branch-consistent AT2 derivatives are undefined at zero eigenvalues or trace",
    )
  positive_eigenvalues, negative_eigenvalues = _macaulay(eigenvalues)
  positive_trace, negative_trace = _macaulay(trace)
  positive_energy = 0.5 * lame_lambda * positive_trace**2 + shear_modulus * np.sum(
    positive_eigenvalues**2, axis=1
  )
  negative_energy = 0.5 * lame_lambda * negative_trace**2 + shear_modulus * np.sum(
    negative_eigenvalues**2, axis=1
  )

  positive_tensor = np.einsum(
    "nij,nj,nkj->nik",
    eigenvectors,
    positive_eigenvalues,
    eigenvectors,
    optimize=True,
  )
  negative_tensor = np.einsum(
    "nij,nj,nkj->nik",
    eigenvectors,
    negative_eigenvalues,
    eigenvectors,
    optimize=True,
  )
  identity = np.eye(2, dtype=np.float64)[None, :, :]
  positive_stress_tensor = (
    lame_lambda * positive_trace[:, None, None] * identity
    + 2.0 * shear_modulus * positive_tensor
  )
  negative_stress_tensor = (
    lame_lambda * negative_trace[:, None, None] * identity
    + 2.0 * shear_modulus * negative_tensor
  )
  positive_stress = _stress_voigt(positive_stress_tensor)
  negative_stress = _stress_voigt(negative_stress_tensor)
  positive_tangent = negative_tangent = None
  if tangent:
    positive_tangent, negative_tangent = _spectral_tangents(
      eigenvalues,
      eigenvectors,
      trace,
      lame_lambda,
      shear_modulus,
    )
  arrays = (
    positive_energy,
    negative_energy,
    positive_stress,
    negative_stress,
    positive_tangent,
    negative_tangent,
    eigenvalues,
  )
  if any(array is not None and not bool(np.isfinite(array).all()) for array in arrays):
    msg = "AT2 spectral response left the finite float64 range"
    raise ValueError(msg)
  return SpectralAT2Response(*arrays)


def frozen_history_phase_potential(
  phase: float,
  history: float,
  critical_energy_release_rate: float,
  length_scale: float,
  area: float = 1.0,
) -> float:
  """Uniform zero-flux AT2 potential with history frozen independently of phase."""
  return area * (
    (1.0 - phase) ** 2 * history
    + critical_energy_release_rate * phase**2 / (2.0 * length_scale)
  )


def frozen_history_phase_solution(
  history: float,
  critical_energy_release_rate: float,
  length_scale: float,
) -> float:
  """Minimize the scalar uniform AT2 potential over positive material parameters."""
  if not np.isfinite(history) or history < 0.0:
    msg = "frozen phase history must be finite and nonnegative"
    raise ValueError(msg)
  if (
    not np.isfinite(critical_energy_release_rate)
    or critical_energy_release_rate <= 0.0
    or not np.isfinite(length_scale)
    or length_scale <= 0.0
  ):
    msg = "phase fracture energy and length scale must be positive and finite"
    raise ValueError(msg)
  return 2.0 * history / (2.0 * history + critical_energy_release_rate / length_scale)
