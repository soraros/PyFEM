# SPDX-License-Identifier: MIT
# Copyright (c) 2011–2026 Joris J.C. Remmers

"""Skorohod-Olevsky viscous sintering model (module-name alias).

MaterialManager resolves a deck's material ``type`` by importing
``pyfem.materials.<type>`` and fetching the class of the same name, so the
module basename must match the class name. The Skorohod-Olevsky law lives in
:mod:`pyfem.materials.SOVS` as class ``SkorohodOlevsky``; this module lets
decks that declare ``type = "SkorohodOlevsky"`` (the class's own documented
name) resolve to it.
"""

from pyfem.materials.SOVS import SkorohodOlevsky

__all__ = ["SkorohodOlevsky"]
