from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import geopandas as gpd
import pandas as pd
from plotnine import (
    aes,
    coord_fixed,
    geom_polygon,
    scale_fill_identity,
    scale_fill_manual,
)
from plotnine.mapping import aes as aes_class

from ggsegpy.join import brain_join
from ggsegpy.palettes import scale_fill_brain
from ggsegpy.themes import theme_brain

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas


class geom_brain:
    """A plotnine-compatible layer for brain atlas visualization.

    Use with ggplot() to create 2D brain atlas plots. The atlas geometry
    is automatically extracted and rendered as polygons.

    Parameters
    ----------
    atlas
        Brain atlas to plot. Defaults to dk() if not specified.
    mapping
        Aesthetic mappings created by aes(). Use fill= for coloring regions.
    data
        DataFrame with values to map onto brain regions.
    hemi
        Hemisphere(s) to show: 'left', 'right', or list of both.
    view
        View(s) to show: 'lateral', 'medial', or list of both.
    position
        Layout arrangement: 'horizontal', 'vertical', or 'stacked'.
    color
        Outline color for regions. Default is 'black'.
    size
        Outline width. Default is 0.1.
    na_fill
        Color for regions with missing data. Default is 'grey'.
    show_legend
        Whether to show the legend. Default is True.
    **kwargs
        Additional arguments passed to geom_polygon.

    Returns
    -------
    geom_brain
        A layer that can be added to a ggplot with +.

    Examples
    --------
    Basic atlas plot:

    >>> from plotnine import ggplot
    >>> from ggsegpy import geom_brain, dk
    >>> ggplot() + geom_brain(atlas=dk())

    Plot with custom data:

    >>> from plotnine import ggplot, aes
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "label": ["lh_precentral", "rh_precentral"],
    ...     "value": [2.5, 2.1]
    ... })
    >>> ggplot(data) + geom_brain(atlas=dk(), mapping=aes(fill="value"))

    Filter by hemisphere and view:

    >>> ggplot() + geom_brain(atlas=dk(), hemi="left", view="lateral")
    """

    def __init__(
        self,
        atlas: BrainAtlas | None = None,
        mapping: aes_class | None = None,
        data: pd.DataFrame | None = None,
        hemi: str | list[str] | None = None,
        view: str | list[str] | None = None,
        position: Literal["horizontal", "vertical", "stacked"] = "horizontal",
        color: str = "black",
        size: float = 0.1,
        na_fill: str = "grey",
        show_legend: bool = True,
        **kwargs: Any,
    ):
        self.atlas = atlas
        self.mapping = mapping
        self.data = data
        self.hemi = hemi
        self.view = view
        self.position = position
        self.color = color
        self.size = size
        self.na_fill = na_fill
        self.show_legend = show_legend
        self.kwargs = kwargs

    def __radd__(self, gg):
        """Allow ggplot() + geom_brain() syntax."""
        return self._build_plot(gg)

    def _build_plot(self, gg):
        if self.atlas is None:
            from ggsegpy.atlases import dk

            atlas = dk()
        else:
            atlas = self.atlas

        plot_data = self._prepare_plot_data(atlas, gg)
        plot_data = _extract_coordinates(plot_data)

        fill_col = self._determine_fill_column(plot_data)

        base_aes = aes(x="x", y="y", group="group_id", fill=fill_col)

        result = gg + geom_polygon(
            data=plot_data,
            mapping=base_aes,
            color=self.color,
            size=self.size,
            **self.kwargs,
        )

        result = result + coord_fixed() + theme_brain()

        if fill_col == "color":
            result = result + scale_fill_identity()
        elif atlas.palette:
            fill_palette = scale_fill_brain(atlas.palette, self.na_fill)
            result = result + scale_fill_manual(
                values=fill_palette, na_value=self.na_fill
            )

        return result

    def _prepare_plot_data(self, atlas: BrainAtlas, gg) -> gpd.GeoDataFrame:
        sf = atlas.data.ggseg.copy()

        user_data = self.data
        if user_data is None and hasattr(gg, "data") and gg.data is not None:
            user_data = gg.data

        if user_data is not None:
            sf = brain_join(user_data, atlas)

        if self.hemi is not None:
            hemis = [self.hemi] if isinstance(self.hemi, str) else self.hemi
            sf = sf[sf["hemi"].isin(hemis)]

        if self.view is not None:
            views = [self.view] if isinstance(self.view, str) else self.view
            sf = sf[sf["view"].isin(views)]

        return sf

    def _determine_fill_column(self, plot_data: pd.DataFrame) -> str:
        if self.mapping is not None and "fill" in self.mapping:
            fill_var = self.mapping["fill"]
            if isinstance(fill_var, str) and fill_var in plot_data.columns:
                return fill_var

        if "color" in plot_data.columns:
            return "color"

        return "region"


def _extract_coordinates(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    import numpy as np

    non_geom_cols = [c for c in gdf.columns if c != "geometry"]
    all_coords = []
    all_groups = []
    all_row_indices = []

    for idx, geom in enumerate(gdf.geometry):
        if geom is None:
            continue

        rings = _get_polygon_coords(geom)
        for ring_idx, ring_coords in enumerate(rings):
            if not ring_coords:
                continue
            coords_arr = np.array(ring_coords)
            n_points = len(coords_arr)
            all_coords.append(coords_arr)
            all_groups.extend([f"{idx}_{ring_idx}"] * n_points)
            all_row_indices.extend([idx] * n_points)

    if not all_coords:
        return pd.DataFrame(columns=non_geom_cols + ["x", "y", "group_id"])

    coords_combined = np.vstack(all_coords)

    result = gdf[non_geom_cols].iloc[all_row_indices].reset_index(drop=True)
    result["x"] = coords_combined[:, 0]
    result["y"] = coords_combined[:, 1]
    result["group_id"] = all_groups

    return result


def _get_polygon_coords(geom) -> list[list[tuple[float, float]]]:
    from shapely.geometry import MultiPolygon, Polygon

    coords = []

    if isinstance(geom, Polygon):
        coords.append(list(geom.exterior.coords))
        for interior in geom.interiors:
            coords.append(list(interior.coords))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            coords.extend(_get_polygon_coords(poly))

    return coords
