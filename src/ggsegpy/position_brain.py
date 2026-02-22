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
    similar to R ggseg's position_brain(). Supports two modes:

    1. Variable-based layout (rows/cols): For cortical atlases, arrange by
       hemisphere and/or view variables.
    2. Grid-based layout (nrow/ncol): For subcortical/tract atlases, distribute
       views into a simple N-row or N-column grid.

    Parameters
    ----------
    rows
        Variable for row faceting: 'hemi', 'view', or None.
        Default is 'hemi' (hemispheres stacked vertically).
        Used for cortical atlases. Ignored if nrow or ncol is set.
    cols
        Variable for column faceting: 'hemi', 'view', or None.
        Default is 'view' (views arranged horizontally).
        Used for cortical atlases. Ignored if nrow or ncol is set.
    nrow
        Number of rows in grid layout. Views are distributed into this many
        rows (column count computed automatically). Useful for subcortical
        and tract atlases.
    ncol
        Number of columns in grid layout. Views are distributed into this
        many columns (row count computed automatically). Useful for subcortical
        and tract atlases.
    views
        List of view names to include, in order. If None, all views are
        included. Useful for subcortical/tract atlases to select specific
        slices (e.g., ['sagittal', 'axial_3', 'coronal_2']).
    spacing
        Relative spacing between panels (fraction of panel size).
        Default is 0.1.

    Examples
    --------
    Default layout for cortical atlas (hemi ~ view):

    >>> ggplot() + geom_brain(atlas=dk(), position=position_brain())

    Views as rows, hemispheres as columns:

    >>> pos = position_brain(rows="view", cols="hemi")
    >>> ggplot() + geom_brain(atlas=dk(), position=pos)

    Simple 2-column layout for subcortical atlas:

    >>> pos = position_brain(ncol=2)
    >>> ggplot() + geom_brain(atlas=aseg(), position=pos)

    Single row with specific views:

    >>> pos = position_brain(nrow=1, views=["sagittal", "axial_3"])
    >>> ggplot() + geom_brain(atlas=aseg(), position=pos)

    Show only lateral views:

    >>> ggplot() + geom_brain(atlas=dk(), position=position_brain(views=["lateral"]))
    """

    rows: Literal["hemi", "view"] | None = "hemi"
    cols: Literal["hemi", "view"] | None = "view"
    nrow: int | None = None
    ncol: int | None = None
    views: list[str] | None = None
    spacing: float = 0.1

    def apply(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Apply positioning to a GeoDataFrame."""
        gdf = gdf.copy()

        if self.views is not None and "view" in gdf.columns:
            gdf = gdf[gdf["view"].isin(self.views)]
            view_order = {v: i for i, v in enumerate(self.views)}
            gdf = gdf.iloc[gdf["view"].map(view_order).argsort()]

        if gdf.empty:
            return gdf

        if self.nrow is not None or self.ncol is not None:
            return self._apply_grid_layout(gdf)

        return self._apply_variable_layout(gdf)

    def _apply_grid_layout(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Distribute panels in a simple grid with nrow/ncol."""
        panels, panel_to_idx = _get_unique_panels(gdf, view_order=self.views)
        n_panels = len(panels)

        if n_panels == 0:
            return gdf

        if self.ncol is not None:
            ncol = self.ncol
        else:
            ncol = int(np.ceil(n_panels / self.nrow))

        panel_bounds = _compute_panel_bounds_grouped(gdf)
        if panel_bounds is None:
            return gdf

        cell_width = panel_bounds["width"] * (1 + self.spacing)
        cell_height = panel_bounds["height"] * (1 + self.spacing)

        x_shifts, y_shifts = _compute_translations(
            gdf, panels, panel_to_idx, ncol, cell_width, cell_height
        )

        translated = [
            affinity.translate(geom, xoff=x_shifts[i], yoff=y_shifts[i])
            for i, geom in enumerate(gdf.geometry)
        ]
        gdf = gdf.set_geometry(gpd.GeoSeries(translated, index=gdf.index, crs=gdf.crs))

        return gdf

    def _apply_variable_layout(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Arrange by hemi/view variables (for cortical atlases)."""
        has_hemi = "hemi" in gdf.columns
        has_view = "view" in gdf.columns

        if not has_hemi and not has_view:
            return gdf

        hemis = _get_ordered_hemis(gdf["hemi"].unique()) if has_hemi else [None]
        views = _get_ordered_views(gdf["view"].unique()) if has_view else [None]

        row_vals = self._get_row_vals(hemis, views, has_hemi, has_view)
        col_vals = self._get_col_vals(hemis, views, has_hemi, has_view)

        panel_bounds = _compute_panel_bounds_grouped(gdf)
        if panel_bounds is None:
            return gdf

        cell_width = panel_bounds["width"] * (1 + self.spacing)
        cell_height = panel_bounds["height"] * (1 + self.spacing)

        x_shifts, y_shifts = _compute_variable_translations(
            gdf,
            row_vals,
            col_vals,
            self.rows,
            self.cols,
            cell_width,
            cell_height,
        )

        translated = [
            affinity.translate(geom, xoff=x_shifts[i], yoff=y_shifts[i])
            for i, geom in enumerate(gdf.geometry)
        ]
        gdf = gdf.set_geometry(gpd.GeoSeries(translated, index=gdf.index, crs=gdf.crs))

        return gdf

    def _get_row_vals(self, hemis, views, has_hemi, has_view):
        if self.rows == "hemi" and has_hemi:
            return hemis
        if self.rows == "view" and has_view:
            return views
        return [None]

    def _get_col_vals(self, hemis, views, has_hemi, has_view):
        if self.cols == "view" and has_view:
            return views
        if self.cols == "hemi" and has_hemi:
            return hemis
        return [None]


def _get_unique_panels(
    gdf: gpd.GeoDataFrame, view_order: list[str] | None = None
) -> tuple[list[tuple], dict[tuple, int]]:
    """Get unique (hemi, view) panel combinations and index mapping."""
    has_hemi = "hemi" in gdf.columns
    has_view = "view" in gdf.columns

    if has_view and has_hemi:
        hemis = _get_ordered_hemis(gdf["hemi"].unique())
        if view_order is not None:
            views = [v for v in view_order if v in gdf["view"].unique()]
        else:
            views = _get_ordered_views(gdf["view"].unique())
        panels = [(h, v) for h in hemis for v in views]
    elif has_view:
        if view_order is not None:
            views = [v for v in view_order if v in gdf["view"].unique()]
        else:
            views = _get_ordered_views(gdf["view"].unique())
        panels = [(None, v) for v in views]
    elif has_hemi:
        hemis = _get_ordered_hemis(gdf["hemi"].unique())
        panels = [(h, None) for h in hemis]
    else:
        panels = [(None, None)]

    panel_to_idx = {p: i for i, p in enumerate(panels)}
    return panels, panel_to_idx


def _compute_panel_bounds_grouped(gdf: gpd.GeoDataFrame) -> dict[str, float] | None:
    """Compute max panel dimensions using groupby."""
    has_hemi = "hemi" in gdf.columns
    has_view = "view" in gdf.columns

    if has_hemi and has_view:
        group_cols = ["hemi", "view"]
    elif has_hemi:
        group_cols = ["hemi"]
    elif has_view:
        group_cols = ["view"]
    else:
        bounds = gdf.total_bounds
        return {"width": bounds[2] - bounds[0], "height": bounds[3] - bounds[1]}

    def get_dimensions(group):
        bounds = group.total_bounds
        return {"width": bounds[2] - bounds[0], "height": bounds[3] - bounds[1]}

    grouped = gdf.groupby(group_cols, observed=True)
    dims = grouped.apply(get_dimensions, include_groups=False)

    if len(dims) == 0:
        return None

    max_width = max(d["width"] for d in dims)
    max_height = max(d["height"] for d in dims)

    if max_width == 0 or max_height == 0:
        return None

    return {"width": max_width, "height": max_height}


def _compute_translations(
    gdf: gpd.GeoDataFrame,
    panels: list[tuple],
    panel_to_idx: dict[tuple, int],
    ncol: int,
    cell_width: float,
    cell_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute translation offsets for each row in the GeoDataFrame."""
    n = len(gdf)
    x_shifts = np.zeros(n)
    y_shifts = np.zeros(n)

    has_hemi = "hemi" in gdf.columns
    has_view = "view" in gdf.columns

    hemi_arr = gdf["hemi"].values if has_hemi else np.array([None] * n)
    view_arr = gdf["view"].values if has_view else np.array([None] * n)

    panel_origins = {}
    for panel in panels:
        hemi, view = panel
        if has_hemi and has_view:
            mask = (gdf["hemi"] == hemi) & (gdf["view"] == view)
        elif has_hemi:
            mask = gdf["hemi"] == hemi
        elif has_view:
            mask = gdf["view"] == view
        else:
            mask = np.ones(n, dtype=bool)

        if mask.any():
            bounds = gdf[mask].total_bounds
            panel_origins[panel] = (bounds[0], bounds[1])

    for i in range(n):
        panel = (hemi_arr[i], view_arr[i])
        if panel not in panel_to_idx:
            continue

        idx = panel_to_idx[panel]
        row_idx = idx // ncol
        col_idx = idx % ncol

        target_x = col_idx * cell_width
        target_y = -row_idx * cell_height

        origin_x, origin_y = panel_origins.get(panel, (0, 0))
        x_shifts[i] = target_x - origin_x
        y_shifts[i] = target_y - origin_y

    return x_shifts, y_shifts


def _compute_variable_translations(
    gdf: gpd.GeoDataFrame,
    row_vals: list,
    col_vals: list,
    rows_var: str | None,
    cols_var: str | None,
    cell_width: float,
    cell_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute translation offsets for variable-based layout."""
    n = len(gdf)
    x_shifts = np.zeros(n)
    y_shifts = np.zeros(n)

    row_to_idx = {v: i for i, v in enumerate(row_vals)}
    col_to_idx = {v: i for i, v in enumerate(col_vals)}

    has_hemi = "hemi" in gdf.columns
    has_view = "view" in gdf.columns

    panel_origins = {}
    for row_val in row_vals:
        for col_val in col_vals:
            mask = np.ones(n, dtype=bool)
            if rows_var == "hemi" and row_val is not None:
                mask &= gdf["hemi"].values == row_val
            elif rows_var == "view" and row_val is not None:
                mask &= gdf["view"].values == row_val

            if cols_var == "view" and col_val is not None:
                mask &= gdf["view"].values == col_val
            elif cols_var == "hemi" and col_val is not None:
                mask &= gdf["hemi"].values == col_val

            if mask.any():
                bounds = gdf[mask].total_bounds
                panel_origins[(row_val, col_val)] = (bounds[0], bounds[1])

    hemi_arr = gdf["hemi"].values if has_hemi else np.array([None] * n)
    view_arr = gdf["view"].values if has_view else np.array([None] * n)

    for i in range(n):
        row_val = None
        col_val = None

        if rows_var == "hemi":
            row_val = hemi_arr[i]
        elif rows_var == "view":
            row_val = view_arr[i]

        if cols_var == "view":
            col_val = view_arr[i]
        elif cols_var == "hemi":
            col_val = hemi_arr[i]

        if row_val not in row_to_idx or col_val not in col_to_idx:
            continue

        row_idx = row_to_idx[row_val]
        col_idx = col_to_idx[col_val]

        target_x = col_idx * cell_width
        target_y = -row_idx * cell_height

        origin_x, origin_y = panel_origins.get((row_val, col_val), (0, 0))
        x_shifts[i] = target_x - origin_x
        y_shifts[i] = target_y - origin_y

    return x_shifts, y_shifts


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


def reposition_brain(
    data: gpd.GeoDataFrame,
    rows: Literal["hemi", "view"] | None = "hemi",
    cols: Literal["hemi", "view"] | None = "view",
    nrow: int | None = None,
    ncol: int | None = None,
    views: list[str] | None = None,
    spacing: float = 0.1,
) -> gpd.GeoDataFrame:
    """Reposition brain atlas geometry for plotting.

    Standalone function for repositioning pre-joined atlas data.
    This is equivalent to R's reposition_brain() function.

    Parameters
    ----------
    data
        GeoDataFrame of brain atlas with geometry, typically from
        brain_join() or atlas.data.ggseg.
    rows
        Variable for row faceting: 'hemi', 'view', or None.
    cols
        Variable for column faceting: 'hemi', 'view', or None.
    nrow
        Number of rows for grid layout.
    ncol
        Number of columns for grid layout.
    views
        List of view names to include, in order.
    spacing
        Relative spacing between panels.

    Returns
    -------
    gpd.GeoDataFrame
        Repositioned GeoDataFrame.

    Examples
    --------
    >>> from ggsegpy import dk, reposition_brain
    >>> atlas = dk()
    >>> data = atlas.data.ggseg
    >>> repositioned = reposition_brain(data, rows="view", cols="hemi")

    >>> # Grid layout for subcortical
    >>> from ggsegpy import aseg
    >>> atlas = aseg()
    >>> repositioned = reposition_brain(atlas.data.ggseg, nrow=2)
    """
    pos = position_brain(
        rows=rows,
        cols=cols,
        nrow=nrow,
        ncol=ncol,
        views=views,
        spacing=spacing,
    )
    return pos.apply(data)
