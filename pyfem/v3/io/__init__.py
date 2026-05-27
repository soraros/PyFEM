"""v3 input adapters."""

from pyfem.v3.io.dat import read_dat_mesh
from pyfem.v3.io.legacy_pro import read_legacy_pro
from pyfem.v3.io.toml import read_problem_toml

__all__ = ["read_dat_mesh", "read_legacy_pro", "read_problem_toml"]
