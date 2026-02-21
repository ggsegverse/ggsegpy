from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import geopandas as gpd
import numpy as np
from shapely import affinity


@dataclass
class position_brain:
    """Position brain regions in a grid layout.

    Arranges brain hemispheres and views in a configurable grid layout,
    similar to R ggseg's position_brain(). By default, creates a standard
    neuroimaging layout with hemispheres as rows and views as columns.

    Parameters
    ----------
    rows
        Variable(s) for row faceting: 'hemi', 'view', or None.
        Default is 'hemi' (hemispheres stacked vertically).
    cols
        Variable(s) for column faceting: 'hemi', 'view', or None.
        Default is 'view' (views arranged horizontally).
    side
        Which side to show: 'left', 'right', or None for both.
        This is a convenience filter applied before positioning.
    spacing
        Relative spacing between panels (fraction of panel size).
        Default is 0.1.

    Examples
    --------
    Default layout (hemi ~ view):

    >>> ggplot() + geom_brain(atlas=dk(), position=position_brain())

    Views as rows, hemispheres as columns:

    >>> pos = position_brain(rows="view", cols="hemi")
    >>> ggplot() + geom_brain(atlas=dk(), position=pos)

    Single column layout:

    >>> pos = position_brain(rows="hemi", cols=None)
    >>> ggplot() + geom_brain(atlas=dk(), position=pos)

    Show only left hemisphere:

    >>> ggplot() + geom_brain(atlas=dk(), position=position_brain(side="left"))

    Notes
    -----
    String shortcuts are also supported for backwards compatibility:
    - "horizontal": equivalent to position_brain(rows=None, cols="view")
    - "vertical": equivalent to position_brain(rows="view", cols=None)
    - "stacked": equivalent to position_brain(rows="hemi", cols="view")
    """

    rows: Literal["hemi", "view"] | None = "hemi"
    cols: Literal["hemi", "view"] | None = "view"
    side: Literal["left", "right"] | None = None
    spacing: float = 0.1

    def apply(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Apply positioning to a GeoDataFrame."""
        gdf = gdf.copy()

        if "view" not in gdf.columns or "hemi" not in gdf.columns:
            return gdf

        if self.side is not None:
            gdf = gdf[gdf["hemi"] == self.side]

        if gdf.empty:
            return gdf

        hemis = _get_ordered_hemis(gdf["hemi"].unique())
        views = _get_ordered_views(gdf["view"].unique())

        row_vals = (
            hemis if self.rows == "hemi" else views if self.rows == "view" else [None]
        )
        col_vals = (
            views if self.cols == "view" else hemis if self.cols == "hemi" else [None]
        )

        panel_bounds = _compute_panel_bounds(gdf, hemis, views)
        if panel_bounds is None:
            return gdf

        panel_width = panel_bounds["width"]
        panel_height = panel_bounds["height"]

        offset_x = panel_width * (1 + self.spacing)
        offset_y = panel_height * (1 + self.spacing)

        for row_idx, row_val in enumerate(row_vals):
            for col_idx, col_val in enumerate(col_vals):
                mask = _build_mask(gdf, self.rows, row_val, self.cols, col_val)
                if not mask.any():
                    continue

                x_shift = col_idx * offset_x
                y_shift = -row_idx * offset_y

                gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].apply(
                    lambda geom: affinity.translate(geom, xoff=x_shift, yoff=y_shift)
                )

        return gdf


def _get_ordered_hemis(hemis: np.ndarray) -> list[str]:
    priority = {"left": 0, "right": 1, "midline": 2, "subcort": 3}
    return sorted(hemis, key=lambda h: priority.get(h, 99))


def _get_ordered_views(views: np.ndarray) -> list[str]:
    priority = {
        "lateral": 0,
        "medial": 1,
        "superior": 2,
        "inferior": 3,
        "anterior": 4,
        "posterior": 5,
    }
    return sorted(views, key=lambda v: priority.get(v, 99))


def _compute_panel_bounds(
    gdf: gpd.GeoDataFrame,
    hemis: list[str],
    views: list[str],
) -> dict[str, float] | None:
    """Compute consistent panel dimensions across all hemi/view combinations."""
    max_width = 0.0
    max_height = 0.0

    for hemi in hemis:
        for view in views:
            mask = (gdf["hemi"] == hemi) & (gdf["view"] == view)
            subset = gdf[mask]
            if subset.empty:
                continue
            bounds = subset.total_bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            max_width = max(max_width, width)
            max_height = max(max_height, height)

    if max_width == 0 or max_height == 0:
        return None

    return {"width": max_width, "height": max_height}


def _build_mask(
    gdf: gpd.GeoDataFrame,
    row_var: str | None,
    row_val: str | None,
    col_var: str | None,
    col_val: str | None,
) -> np.ndarray:
    """Build a boolean mask for the current row/col combination."""
    mask = np.ones(len(gdf), dtype=bool)

    if row_var == "hemi" and row_val is not None:
        mask &= gdf["hemi"] == row_val
    elif row_var == "view" and row_val is not None:
        mask &= gdf["view"] == row_val

    if col_var == "view" and col_val is not None:
        mask &= gdf["view"] == col_val
    elif col_var == "hemi" and col_val is not None:
        mask &= gdf["hemi"] == col_val

    return mask


def resolve_position(
    position: str | position_brain,
) -> position_brain:
    """Convert string shortcuts to position_brain objects."""
    if isinstance(position, position_brain):
        return position

    if position == "horizontal":
        return position_brain(rows=None, cols="view")
    elif position == "vertical":
        return position_brain(rows="view", cols=None)
    elif position == "stacked":
        return position_brain(rows="hemi", cols="view")
    else:
        return position_brain()
