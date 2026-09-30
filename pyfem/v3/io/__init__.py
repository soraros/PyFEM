"""v3 input adapters."""

from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.io.legacy_deck import (
  CompiledDeck,
  ConvertedDeck,
  DeckConversionError,
  DeckRun,
  DeckSolverSettings,
  compile_deck,
  read_legacy_deck,
  run_deck,
)
from pyfem.v3.io.legacy_pro import read_legacy_pro
from pyfem.v3.io.toml import read_problem_toml

__all__ = [
  "CompiledDeck",
  "ConvertedDeck",
  "DeckConversionError",
  "DeckRun",
  "DeckSolverSettings",
  "compile_deck",
  "read_dat_mesh",
  "read_legacy_deck",
  "read_legacy_pro",
  "read_problem_toml",
  "run_deck",
]
