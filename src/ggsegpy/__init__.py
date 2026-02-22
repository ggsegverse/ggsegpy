"""
ggsegpy: Brain atlas visualization in Python.

A Python port of the R ggseg ecosystem for visualizing brain atlases
in 2D (using plotnine) and 3D (using plotly).
"""

from __future__ import annotations

from ggsegpy.atlas import (
    BrainAtlas,
    CorticalAtlas,
    SubcorticalAtlas,
    TractAtlas,
)
from ggsegpy.atlas_utils import (
    atlas_core_add,
    atlas_labels,
    atlas_meshes,
    atlas_palette,
    atlas_region_contextual,
    atlas_region_keep,
    atlas_region_remove,
    atlas_region_rename,
    atlas_regions,
    atlas_sf,
    atlas_type,
    atlas_vertices,
    atlas_view_gather,
    atlas_view_keep,
    atlas_view_remove,
    atlas_view_remove_region,
    atlas_view_remove_small,
    atlas_view_reorder,
    atlas_views,
)
from ggsegpy.atlases import aseg, dk, tracula
from ggsegpy.data import (
    AtlasData,
    CorticalData,
    SubcorticalData,
    SurfaceMesh,
    TractData,
)
from ggsegpy.geom_brain import annotate_brain, geom_brain
from ggsegpy.join import brain_join
from ggsegpy.palettes import (
    generate_palette,
    scale_color_brain_manual,
    scale_colour_brain_manual,
    scale_fill_brain,
    scale_fill_brain_manual,
)
from ggsegpy.plot3d import (
    BrainFigure,
    add_atlas,
    add_glassbrain,
    ggseg3d,
    pan_camera,
    remove_legend,
    set_background,
    set_flat_shading,
    set_hover,
    set_legend,
    set_opacity,
    set_orthographic,
    set_surface_color,
)
from ggsegpy.position_brain import position_brain, reposition_brain
from ggsegpy.themes import (
    theme_brain,
    theme_brain_void,
    theme_custombrain,
    theme_darkbrain,
)
from ggsegpy.validation import (
    is_cortical_atlas,
    is_ggseg_atlas,
    is_subcortical_atlas,
    is_tract_atlas,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "add_atlas",
    "add_glassbrain",
    "annotate_brain",
    "aseg",
    "atlas_core_add",
    "atlas_labels",
    "atlas_meshes",
    "atlas_palette",
    "atlas_region_contextual",
    "atlas_region_keep",
    "atlas_region_remove",
    "atlas_region_rename",
    "atlas_regions",
    "atlas_sf",
    "atlas_type",
    "atlas_vertices",
    "atlas_view_gather",
    "atlas_view_keep",
    "atlas_view_remove",
    "atlas_view_remove_region",
    "atlas_view_remove_small",
    "atlas_view_reorder",
    "atlas_views",
    "AtlasData",
    "brain_join",
    "BrainAtlas",
    "BrainFigure",
    "CorticalAtlas",
    "CorticalData",
    "dk",
    "generate_palette",
    "geom_brain",
    "ggseg3d",
    "is_cortical_atlas",
    "is_ggseg_atlas",
    "is_subcortical_atlas",
    "is_tract_atlas",
    "pan_camera",
    "position_brain",
    "remove_legend",
    "reposition_brain",
    "scale_color_brain_manual",
    "scale_colour_brain_manual",
    "scale_fill_brain",
    "scale_fill_brain_manual",
    "set_background",
    "set_flat_shading",
    "set_hover",
    "set_legend",
    "set_opacity",
    "set_orthographic",
    "set_surface_color",
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
