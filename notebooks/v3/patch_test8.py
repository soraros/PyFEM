# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PatchTest8 — authored v3 linear-static solve
#
# Requires Python 3.13+ and `uv sync --group v3`.

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

if sys.version_info < (3, 13):
  raise RuntimeError("This notebook requires Python 3.13+")

from pyfem.v3 import authoring
from pyfem.v3.io.legacy_deck import read_legacy_deck

ROOT = Path.cwd()
if not (ROOT / "skims").exists():
  ROOT = ROOT.parents[1]

deck = read_legacy_deck(ROOT / "skims/patch_test8/skim.pro")
system = authoring.compile(deck.model, deck.registry)
session = authoring.nonlinear_static(
  system,
  constraints=deck.program.constraints,
  loads=deck.program.loads,
  coordinates=deck.program.coordinates,
)
result = session.run({"load": 0.0}, {"load": 1.0})
assert result.status is authoring.DriverStatus.COMPLETED

# %% [markdown]
# The prescribed boundary displacements come from the analytic patch field
# \(u_x = 10^{-3}x + 5\times10^{-4}y\) and
# \(u_y = 5\times10^{-4}x + 10^{-3}y\). The committed solution must recover
# that field at every node, including the free interior nodes.

# %%
coordinates = np.asarray([node.coordinates for node in deck.model.mesh.nodes])
node_rows = {node.id: row for row, node in enumerate(deck.model.mesh.nodes)}
displacements = np.empty((len(coordinates), 2))
state = session.accepted_coefficients()
for row, (_, node_id, component) in enumerate(session.system.spaces[0].coefficient_ids):
  component_index = {"x": 0, "y": 1}[component]
  displacements[node_rows[node_id], component_index] = state[row]

expected = np.column_stack(
  (
    1.0e-3 * coordinates[:, 0] + 5.0e-4 * coordinates[:, 1],
    5.0e-4 * coordinates[:, 0] + 1.0e-3 * coordinates[:, 1],
  )
)
np.testing.assert_allclose(displacements, expected, rtol=0.0, atol=1.0e-12)

# %%
try:
  get_ipython().run_line_magic("matplotlib", "widget")
except Exception:
  plt.switch_backend("Agg")

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.tripcolor(
  coordinates[:, 0],
  coordinates[:, 1],
  displacements[:, 0],
  shading="gouraud",
)
fig.colorbar(sc, ax=ax, label=r"$u_x$")
ax.set_aspect("equal")
ax.set_title("PatchTest8 — committed $u_x$ displacement (v3)")
fig.tight_layout()
plt.show()
