"""
ggsegpy: Brain atlas visualization in Python.

A Python port of the R ggseg ecosystem for visualizing brain atlases
in 2D (using plotnine) and 3D (using plotly).
"""

from __future__ import annotations

from ggsegpy.atlas import BrainAtlas, CorticalAtlas, SubcorticalAtlas, TractAtlas
from ggsegpy.atlases import aseg, dk, tracula
from ggsegpy.data import AtlasData, CorticalData, SubcorticalData, SurfaceMesh, TractData
from ggsegpy.geom_brain import geom_brain
from ggsegpy.join import brain_join
from ggsegpy.palettes import generate_palette, scale_fill_brain
from ggsegpy.plot3d import (
    add_glassbrain,
    ggseg3d,
    pan_camera,
    remove_legend,
    set_background,
    set_hover,
    set_legend,
)
from ggsegpy.position_brain import position_brain
from ggsegpy.themes import theme_brain, theme_brain_void, theme_custombrain, theme_darkbrain

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "add_glassbrain",
    "aseg",
    "AtlasData",
    "brain_join",
    "BrainAtlas",
    "CorticalAtlas",
    "CorticalData",
    "dk",
    "generate_palette",
    "geom_brain",
    "ggseg3d",
    "pan_camera",
    "position_brain",
    "remove_legend",
    "scale_fill_brain",
    "set_background",
    "set_hover",
    "set_legend",
    "SubcorticalAtlas",
    "SubcorticalData",
    "SurfaceMesh",
    "theme_brain",
    "theme_brain_void",
    "theme_custombrain",
    "theme_darkbrain",
    "TractAtlas",
    "TractData",
    "tracula",
]
