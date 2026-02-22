from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
from ggsegpy.position_brain import position_brain
from ggsegpy.themes import theme_brain

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas


class _BrainLayers:
    """Container for brain layers that defers data preparation until added to ggplot."""

    def __init__(
        self,
        atlas,
        mapping,
        hemi,
        view,
        position,
        color,
        size,
        na_fill,
        show_legend,
        kwargs,
    ):
        self.atlas = atlas
        self.mapping = mapping
        self.hemi = hemi
        self.view = view
        self.position = position
        self.color = color
        self.size = size
        self.na_fill = na_fill
        self.show_legend = show_legend
        self.kwargs = kwargs
        self._extra_layers = []

    @property
    def layers(self):
        return self._build_layers(None)

    def _build_layers(self, gg_data):
        atlas = self.atlas
        if atlas is None:
            from ggsegpy.atlases import dk

            atlas = dk()

        sf = atlas.data.ggseg.copy()

        explicit_data = getattr(self, "_explicit_data", None)
        if explicit_data is not None:
            sf = brain_join(explicit_data, atlas)
        elif gg_data is not None and len(gg_data) > 0:
            sf = brain_join(gg_data, atlas)

        if self.hemi is not None:
            hemis = [self.hemi] if isinstance(self.hemi, str) else self.hemi
            sf = sf[sf["hemi"].isin(hemis)]

        if self.view is not None:
            views = [self.view] if isinstance(self.view, str) else self.view
            sf = sf[sf["view"].isin(views)]

        pos = self.position if self.position is not None else position_brain()
        sf = pos.apply(sf)

        plot_data = _extract_coordinates(sf)
        fill_col = _determine_fill_column(plot_data, self.mapping)

        base_aes = aes(x="x", y="y", group="group_id", fill=fill_col)

        layers = [
            geom_polygon(
                data=plot_data,
                mapping=base_aes,
                color=self.color,
                size=self.size,
                show_legend=self.show_legend,
                **self.kwargs,
            ),
            coord_fixed(),
            theme_brain(),
        ]

        if fill_col == "color":
            layers.append(scale_fill_identity())
        elif fill_col == "region" and atlas.palette:
            fill_palette = scale_fill_brain(atlas.palette, self.na_fill)
            layers.append(scale_fill_manual(values=fill_palette, na_value=self.na_fill))

        return layers + self._extra_layers

    def __radd__(self, gg):
        gg_data = getattr(gg, "data", None)
        layers = self._build_layers(gg_data)
        for layer in layers:
            layer.__radd__(gg)

    def __add__(self, other):
        self._extra_layers.append(other)
        return self


def geom_brain(
    atlas: BrainAtlas | None = None,
    mapping: aes_class | None = None,
    data: pd.DataFrame | None = None,
    hemi: str | list[str] | None = None,
    view: str | list[str] | None = None,
    position: position_brain | None = None,
    color: str = "black",
    size: float = 0.1,
    na_fill: str = "grey",
    show_legend: bool = True,
    **kwargs: Any,
) -> _BrainLayers:
    """Create a plotnine-compatible layer for brain atlas visualization.

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
        Layout arrangement as a position_brain() object. Controls how
        hemispheres and views are arranged in the plot grid.
        Default is position_brain() (hemispheres as rows, views as columns).
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
    _BrainLayers
        A set of layers that can be added to a ggplot with +.

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
    brain_layers = _BrainLayers(
        atlas=atlas,
        mapping=mapping,
        hemi=hemi,
        view=view,
        position=position,
        color=color,
        size=size,
        na_fill=na_fill,
        show_legend=show_legend,
        kwargs=kwargs,
    )

    if data is not None:
        brain_layers._explicit_data = data

    return brain_layers


def _determine_fill_column(plot_data: pd.DataFrame, mapping: aes_class | None) -> str:
    if mapping is not None and "fill" in mapping:
        fill_var = mapping["fill"]
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


def annotate_brain(
    atlas: BrainAtlas | None = None,
    position: position_brain | None = None,
    hemi: str | list[str] | None = None,
    view: str | list[str] | None = None,
    size: float = 3,
    colour: str = "grey30",
    family: str = "mono",
    nudge_y: float = 0,
    **kwargs: Any,
):
    """Add view labels to brain atlas plots.

    Annotates each brain view with a text label positioned above the
    view's bounding box. For cortical atlases, labels show hemisphere
    and view (e.g., "left lateral"). For subcortical and tract atlases,
    labels show the view name directly.

    Labels respect the repositioning done by position_brain(), so the
    same position argument should be passed to both geom_brain() and
    annotate_brain().

    Parameters
    ----------
    atlas
        A brain atlas object (e.g. dk(), aseg()).
    position
        A position_brain() object matching the one used in geom_brain().
    hemi
        Hemisphere(s) to include. If None, all hemispheres are included.
    view
        View(s) to include. If None, all views are included.
    size
        Text size in points. Default is 3.
    colour
        Text colour. Default is 'grey30'.
    family
        Font family. Default is 'mono'.
    nudge_y
        Additional vertical offset for labels. Default is 0.
    **kwargs
        Additional arguments passed to annotate().

    Returns
    -------
    list
        A list of plotnine annotation layers.

    Examples
    --------
    >>> from plotnine import ggplot
    >>> from ggsegpy import geom_brain, annotate_brain, dk, position_brain
    >>> pos = position_brain()
    >>> p = (
    ...     ggplot()
    ...     + geom_brain(atlas=dk(), position=pos, show_legend=False)
    ...     + annotate_brain(atlas=dk(), position=pos)
    ... )
    """
    from plotnine import annotate

    if atlas is None:
        from ggsegpy.atlases import dk

        atlas = dk()

    sf = atlas.data.ggseg.copy()

    if hemi is not None:
        hemis = [hemi] if isinstance(hemi, str) else hemi
        sf = sf[sf["hemi"].isin(hemis)]

    if view is not None:
        views = [view] if isinstance(view, str) else view
        sf = sf[sf["view"].isin(views)]

    pos = position if position is not None else position_brain()
    repositioned = pos.apply(sf)

    label_df = _compute_label_positions(repositioned, atlas.type)

    overall_bounds = repositioned.total_bounds
    y_range = overall_bounds[3] - overall_bounds[1]
    label_df["y"] = label_df["y"] + y_range * 0.02 + nudge_y

    annotations = []
    for _, row in label_df.iterrows():
        annotations.append(
            annotate(
                "text",
                x=row["x"],
                y=row["y"],
                label=row["label"],
                size=size,
                color=colour,
                family=family,
                **kwargs,
            )
        )

    return annotations


def _compute_label_positions(
    repositioned: gpd.GeoDataFrame,
    atlas_type: str,
) -> pd.DataFrame:
    """Compute label text and positions from repositioned brain data."""
    has_hemi = "hemi" in repositioned.columns
    has_view = "view" in repositioned.columns

    if atlas_type == "cortical" and has_hemi and has_view:
        groups = repositioned.groupby(["hemi", "view"], observed=True)

        def make_label(group):
            hemi_val = group["hemi"].iloc[0]
            view_val = group["view"].iloc[0]
            return f"{hemi_val} {view_val}"
    else:
        if has_view:
            groups = repositioned.groupby("view", observed=True)

            def make_label(group):
                return group["view"].iloc[0]
        elif has_hemi:
            groups = repositioned.groupby("hemi", observed=True)

            def make_label(group):
                return group["hemi"].iloc[0]
        else:
            return pd.DataFrame(columns=["x", "y", "label"])

    labels = []
    for _, group in groups:
        bounds = group.total_bounds
        x = (bounds[0] + bounds[2]) / 2
        y = bounds[3]
        label = make_label(group)
        labels.append({"x": x, "y": y, "label": label})

    return pd.DataFrame(labels)


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
