from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Callable

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas
    from ggsegpy.data import AtlasData


def _get_hemi_from_label(label: str | None) -> str:
    """Extract hemisphere from label prefix."""
    if pd.isna(label):
        return ""
    if label.startswith(("lh_", "lh.")):
        return "left"
    if label.startswith(("rh_", "rh.")):
        return "right"
    return ""


def _has_sf_data(atlas: BrainAtlas) -> bool:
    """Check if atlas has 2D geometry data."""
    return hasattr(atlas.data, "ggseg") and atlas.data.ggseg is not None


def _has_3d_data(atlas: BrainAtlas) -> bool:
    """Check if atlas has 3D geometry data."""
    return hasattr(atlas.data, "ggseg3d") and atlas.data.ggseg3d is not None


def _clone_atlas(
    atlas: BrainAtlas,
    core: pd.DataFrame | None = None,
    data: AtlasData | None = None,
    palette: dict[str, str] | None = None,
) -> BrainAtlas:
    """Clone atlas with optional overrides."""
    return atlas.__class__(
        atlas=atlas.atlas,
        type=atlas.type,
        core=core if core is not None else atlas.core,
        data=data if data is not None else atlas.data,
        palette=palette if palette is not None else atlas.palette,
    )


def _validate_match_on(match_on: str) -> None:
    """Validate match_on parameter."""
    if match_on not in ("region", "label"):
        raise ValueError("match_on must be 'region' or 'label'")


def _validate_gap(gap: float) -> None:
    """Validate gap parameter is in reasonable range."""
    if not 0 <= gap <= 2.0:
        raise ValueError("gap must be between 0 and 2.0")


def _escape_and_join_patterns(patterns: list[str]) -> str:
    """Escape regex special characters and join with OR."""
    return "|".join(re.escape(p) for p in patterns)


def atlas_regions(atlas: BrainAtlas) -> list[str]:
    """Extract unique region names from an atlas.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    list[str]
        Sorted list of unique region names.

    Examples
    --------
    >>> from ggsegpy import dk, atlas_regions
    >>> atlas_regions(dk())[:3]
    ['banks of superior temporal sulcus', 'caudal anterior cingulate', ...]
    """
    regions = atlas.core["region"].dropna().unique().tolist()
    return sorted(regions)


def atlas_labels(atlas: BrainAtlas) -> list[str]:
    """Extract unique labels from an atlas.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    list[str]
        Sorted list of unique labels.

    Examples
    --------
    >>> from ggsegpy import dk, atlas_labels
    >>> atlas_labels(dk())[:3]
    ['lh_bankssts', 'lh_caudalanteriorcingulate', ...]
    """
    labels = atlas.core["label"].dropna().unique().tolist()
    return sorted(labels)


def atlas_type(atlas: BrainAtlas) -> str:
    """Get atlas type.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    str
        One of 'cortical', 'subcortical', or 'tract'.
    """
    return atlas.type


def atlas_views(atlas: BrainAtlas) -> list[str] | None:
    """Get available views in atlas.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    list[str] | None
        List of view names, or None if no 2D data.

    Examples
    --------
    >>> from ggsegpy import dk, atlas_views
    >>> atlas_views(dk())
    ['lateral', 'medial', 'inferior', 'superior']
    """
    if not _has_sf_data(atlas):
        return None
    return atlas.data.ggseg["view"].unique().tolist()


def atlas_palette(atlas: BrainAtlas) -> dict[str, str]:
    """Get atlas color palette.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    dict[str, str]
        Mapping of labels to hex colors.
    """
    return atlas.palette.copy()


def atlas_sf(atlas: BrainAtlas) -> gpd.GeoDataFrame:
    """Get atlas 2D geometry data.

    Returns sf data joined with core region info and palette colors.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame ready for plotting.

    Raises
    ------
    ValueError
        If atlas has no 2D geometry.
    """
    if not _has_sf_data(atlas):
        raise ValueError("Atlas does not contain sf geometry for 2D rendering")

    geometry_df = atlas.data.ggseg.copy()

    for col in ["hemi", "region"]:
        if col in geometry_df.columns:
            del geometry_df[col]

    result = geometry_df.merge(atlas.core, on="label", how="left")

    if atlas.palette:
        result["colour"] = result["label"].map(atlas.palette)

    return gpd.GeoDataFrame(result, geometry="geometry")


def atlas_vertices(atlas: BrainAtlas) -> pd.DataFrame:
    """Get atlas vertex data for 3D cortical rendering.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    pd.DataFrame
        DataFrame with vertex indices and region info.

    Raises
    ------
    ValueError
        If atlas has no vertex data.
    """
    if not _has_3d_data(atlas):
        raise ValueError("Atlas does not contain vertices for 3D rendering")

    if "vertex_indices" not in atlas.data.ggseg3d.columns:
        raise ValueError("Atlas does not contain vertex_indices column")

    result = atlas.data.ggseg3d.merge(atlas.core, on="label", how="left")

    if atlas.palette:
        result["colour"] = result["label"].map(atlas.palette)

    return result


def atlas_meshes(atlas: BrainAtlas) -> pd.DataFrame:
    """Get atlas mesh data for 3D subcortical/tract rendering.

    Parameters
    ----------
    atlas
        Brain atlas object.

    Returns
    -------
    pd.DataFrame
        DataFrame with mesh vertices/faces and region info.

    Raises
    ------
    ValueError
        If atlas has no mesh data.
    """
    if not _has_3d_data(atlas):
        raise ValueError("Atlas does not contain meshes for 3D rendering")

    result = atlas.data.ggseg3d.merge(atlas.core, on="label", how="left")

    if atlas.palette:
        result["colour"] = result["label"].map(atlas.palette)

    return result


def atlas_region_remove(
    atlas: BrainAtlas,
    pattern: str,
    match_on: str = "region",
) -> BrainAtlas:
    """Remove regions matching a pattern.

    Completely removes regions from core, palette, sf, and 3D data.

    Parameters
    ----------
    atlas
        Brain atlas object.
    pattern
        Regex pattern to match (case-insensitive).
    match_on
        Column to match: 'region' or 'label'.

    Returns
    -------
    BrainAtlas
        Modified atlas copy.

    Examples
    --------
    >>> from ggsegpy import dk, atlas_region_remove
    >>> atlas = dk()
    >>> filtered = atlas_region_remove(atlas, "banks")
    """
    _validate_match_on(match_on)

    regex = re.compile(pattern, re.IGNORECASE)
    match_col = atlas.core[match_on]

    keep_mask = ~match_col.str.contains(regex, na=False)
    labels_to_remove = set(atlas.core.loc[~keep_mask, "label"])

    new_core = atlas.core[keep_mask].copy()
    new_palette = {k: v for k, v in atlas.palette.items() if k not in labels_to_remove}
    new_data = _rebuild_data_without_labels(atlas, labels_to_remove, remove_sf=True)

    return _clone_atlas(atlas, core=new_core, data=new_data, palette=new_palette)


def atlas_region_contextual(
    atlas: BrainAtlas,
    pattern: str,
    match_on: str = "region",
) -> BrainAtlas:
    """Keep region geometry but remove from core/palette.

    Context geometries render grey and don't appear in legends.

    Parameters
    ----------
    atlas
        Brain atlas object.
    pattern
        Regex pattern to match (case-insensitive).
    match_on
        Column to match: 'region' or 'label'.

    Returns
    -------
    BrainAtlas
        Modified atlas copy.
    """
    _validate_match_on(match_on)

    regex = re.compile(pattern, re.IGNORECASE)
    match_col = atlas.core[match_on]

    keep_mask = ~match_col.str.contains(regex, na=False)
    labels_to_remove = set(atlas.core.loc[~keep_mask, "label"])

    new_core = atlas.core[keep_mask].copy()
    new_palette = {k: v for k, v in atlas.palette.items() if k not in labels_to_remove}
    new_data = _rebuild_data_without_labels(atlas, labels_to_remove, remove_sf=False)

    return _clone_atlas(atlas, core=new_core, data=new_data, palette=new_palette)


def atlas_region_rename(
    atlas: BrainAtlas,
    pattern: str,
    replacement: str | Callable[[str], str],
) -> BrainAtlas:
    """Rename regions matching a pattern.

    Only affects the 'region' column, not 'label'.

    Parameters
    ----------
    atlas
        Brain atlas object.
    pattern
        Regex pattern to match (case-insensitive).
    replacement
        Replacement string or function that takes matched names.

    Returns
    -------
    BrainAtlas
        Modified atlas copy.

    Examples
    --------
    >>> from ggsegpy import dk, atlas_region_rename
    >>> atlas = dk()
    >>> renamed = atlas_region_rename(atlas, "superior", "sup.")
    """
    new_core = atlas.core.copy()
    regex = re.compile(pattern, re.IGNORECASE)

    if callable(replacement):
        mask = new_core["region"].str.contains(regex, na=False)
        new_core.loc[mask, "region"] = new_core.loc[mask, "region"].apply(replacement)
    else:
        new_core["region"] = new_core["region"].str.replace(
            regex, replacement, regex=True
        )

    return _clone_atlas(atlas, core=new_core)


def atlas_region_keep(
    atlas: BrainAtlas,
    pattern: str,
    match_on: str = "region",
) -> BrainAtlas:
    """Keep only regions matching a pattern.

    Non-matching regions are removed from core, palette, and 3D data,
    but sf geometry is preserved for surface continuity.

    Parameters
    ----------
    atlas
        Brain atlas object.
    pattern
        Regex pattern to match (case-insensitive).
    match_on
        Column to match: 'region' or 'label'.

    Returns
    -------
    BrainAtlas
        Modified atlas copy.

    Examples
    --------
    >>> from ggsegpy import dk, atlas_region_keep
    >>> atlas = dk()
    >>> frontal = atlas_region_keep(atlas, "frontal")
    """
    _validate_match_on(match_on)

    regex = re.compile(pattern, re.IGNORECASE)
    match_col = atlas.core[match_on]

    keep_mask = match_col.str.contains(regex, na=False)
    labels_to_keep = set(atlas.core.loc[keep_mask, "label"])
    labels_to_remove = set(atlas.core["label"]) - labels_to_keep

    new_core = atlas.core[keep_mask].copy()
    new_palette = {k: v for k, v in atlas.palette.items() if k in labels_to_keep}
    new_data = _rebuild_data_without_labels(atlas, labels_to_remove, remove_sf=False)

    return _clone_atlas(atlas, core=new_core, data=new_data, palette=new_palette)


def atlas_core_add(
    atlas: BrainAtlas,
    data: pd.DataFrame,
    by: str = "region",
) -> BrainAtlas:
    """Join additional metadata columns to atlas core.

    Parameters
    ----------
    atlas
        Brain atlas object.
    data
        DataFrame with metadata to join.
    by
        Column to join by. Default is 'region'.

    Returns
    -------
    BrainAtlas
        Modified atlas copy.

    Examples
    --------
    >>> import pandas as pd
    >>> from ggsegpy import dk, atlas_core_add
    >>> atlas = dk()
    >>> meta = pd.DataFrame({"region": ["precentral"], "lobe": ["frontal"]})
    >>> enriched = atlas_core_add(atlas, meta, by="region")
    """
    new_core = atlas.core.merge(data, on=by, how="left")
    return _clone_atlas(atlas, core=new_core)


def atlas_view_remove(atlas: BrainAtlas, views: str | list[str]) -> BrainAtlas:
    """Remove views matching pattern from sf data.

    Parameters
    ----------
    atlas
        Brain atlas object.
    views
        View name(s) to remove. Matched exactly (case-insensitive).

    Returns
    -------
    BrainAtlas
        Modified atlas copy.
    """
    if not _has_sf_data(atlas):
        warnings.warn("Atlas has no sf data, nothing to remove")
        return atlas

    if isinstance(views, str):
        views = [views]

    pattern = _escape_and_join_patterns(views)
    regex = re.compile(pattern, re.IGNORECASE)

    keep_mask = ~atlas.data.ggseg["view"].str.contains(regex, na=False)
    new_geometry = atlas.data.ggseg[keep_mask].copy()

    if len(new_geometry) == 0:
        warnings.warn("All views removed, sf data will be empty")

    new_data = _rebuild_atlas_data(atlas, new_geometry)
    return _clone_atlas(atlas, data=new_data)


def atlas_view_keep(atlas: BrainAtlas, views: str | list[str]) -> BrainAtlas:
    """Keep only views matching pattern.

    Parameters
    ----------
    atlas
        Brain atlas object.
    views
        View name(s) to keep. Matched exactly (case-insensitive).

    Returns
    -------
    BrainAtlas
        Modified atlas copy.
    """
    if not _has_sf_data(atlas):
        warnings.warn("Atlas has no sf data, nothing to keep")
        return atlas

    if isinstance(views, str):
        views = [views]

    pattern = _escape_and_join_patterns(views)
    regex = re.compile(pattern, re.IGNORECASE)

    keep_mask = atlas.data.ggseg["view"].str.contains(regex, na=False)
    new_geometry = atlas.data.ggseg[keep_mask].copy()

    if len(new_geometry) == 0:
        warnings.warn("No views matched pattern, sf data will be empty")

    new_data = _rebuild_atlas_data(atlas, new_geometry)
    return _clone_atlas(atlas, data=new_data)


def atlas_view_remove_region(
    atlas: BrainAtlas,
    pattern: str,
    match_on: str = "label",
    views: str | list[str] | None = None,
) -> BrainAtlas:
    """Remove specific region geometry from sf data only.

    Core, palette, and 3D data are unchanged. Use this to clean up
    2D projections without affecting the underlying atlas structure.

    Parameters
    ----------
    atlas
        Brain atlas object.
    pattern
        Regex pattern to match (case-insensitive).
    match_on
        Column to match: 'label' or 'region'.
    views
        Optional view name(s) to scope the removal. If None, applies to all views.

    Returns
    -------
    BrainAtlas
        Modified atlas copy.
    """
    if not _has_sf_data(atlas):
        warnings.warn("Atlas has no sf data, nothing to remove")
        return atlas

    _validate_match_on(match_on)

    geometry_df = atlas.data.ggseg
    regex = re.compile(pattern, re.IGNORECASE)

    if match_on == "region":
        match_col = atlas.core[match_on]
        hit = match_col.str.contains(regex, na=False)
        labels_to_remove = set(atlas.core.loc[hit, "label"])
        is_match = geometry_df["label"].isin(labels_to_remove)
    else:
        is_match = geometry_df["label"].str.contains(regex, na=False)

    if views is not None:
        if isinstance(views, str):
            views = [views]
        view_pattern = _escape_and_join_patterns(views)
        view_regex = re.compile(view_pattern, re.IGNORECASE)
        in_view = geometry_df["view"].str.contains(view_regex, na=False)
        is_match = is_match & in_view

    is_match = is_match.fillna(False)
    new_geometry = geometry_df[~is_match].copy()

    if len(new_geometry) == 0:
        warnings.warn("All region geometries removed, sf data will be empty")

    new_data = _rebuild_atlas_data(atlas, new_geometry)
    return _clone_atlas(atlas, data=new_data)


def atlas_view_remove_small(
    atlas: BrainAtlas,
    min_area: float,
    views: str | list[str] | None = None,
) -> BrainAtlas:
    """Remove region geometries below a minimum area threshold.

    Context geometries (labels not in core) are never removed.
    Useful for cleaning up small polygon fragments from 2D projections.

    Parameters
    ----------
    atlas
        Brain atlas object.
    min_area
        Minimum polygon area to keep.
    views
        Optional view name(s) to scope the removal. If None, applies to all views.

    Returns
    -------
    BrainAtlas
        Modified atlas copy.
    """
    if not _has_sf_data(atlas):
        warnings.warn("Atlas has no sf data, nothing to remove")
        return atlas

    geometry_df = atlas.data.ggseg
    areas = geometry_df.geometry.area

    core_labels = set(atlas.core["label"])
    is_context = geometry_df["label"].isna() | ~geometry_df["label"].isin(core_labels)
    is_small = (areas < min_area) & ~is_context

    if views is not None:
        if isinstance(views, str):
            views = [views]
        view_pattern = _escape_and_join_patterns(views)
        view_regex = re.compile(view_pattern, re.IGNORECASE)
        in_view = geometry_df["view"].str.contains(view_regex, na=False)
        is_small = is_small & in_view

    n_removed = is_small.sum()
    if n_removed > 0:
        warnings.warn(f"Removed {n_removed} geometries below area {min_area}")

    new_geometry = geometry_df[~is_small].copy()
    new_data = _rebuild_atlas_data(atlas, new_geometry)
    return _clone_atlas(atlas, data=new_data)


def atlas_view_gather(
    atlas: BrainAtlas,
    gap: float = 0.15,
) -> BrainAtlas:
    """Reposition views to close gaps after view removal.

    Centers each view group and arranges them horizontally with
    specified gap spacing.

    Parameters
    ----------
    atlas
        Brain atlas object.
    gap
        Proportional gap between views (default 0.15 = 15% of max width).
        Must be between 0 and 2.0.

    Returns
    -------
    BrainAtlas
        Modified atlas copy with repositioned geometry.
    """
    if not _has_sf_data(atlas):
        warnings.warn("Atlas has no sf data")
        return atlas

    _validate_gap(gap)

    new_geometry = _reposition_views(atlas.data.ggseg, atlas_type=atlas.type, gap=gap)
    new_data = _rebuild_atlas_data(atlas, new_geometry)
    return _clone_atlas(atlas, data=new_data)


def atlas_view_reorder(
    atlas: BrainAtlas,
    order: list[str],
    gap: float = 0.15,
) -> BrainAtlas:
    """Reorder views and reposition.

    Views not in order are appended at end.

    Parameters
    ----------
    atlas
        Brain atlas object.
    order
        Desired view order as list of view names.
    gap
        Proportional gap between views (default 0.15 = 15% of max width).
        Must be between 0 and 2.0.

    Returns
    -------
    BrainAtlas
        Modified atlas copy with reordered and repositioned geometry.
    """
    if not _has_sf_data(atlas):
        warnings.warn("Atlas has no sf data")
        return atlas

    _validate_gap(gap)

    geometry_df = atlas.data.ggseg
    current_views = geometry_df["view"].unique().tolist()

    missing_from_order = [view for view in current_views if view not in order]
    if missing_from_order:
        order = list(order) + missing_from_order

    order = [view for view in order if view in current_views]

    if not order:
        warnings.warn("No matching views found in order specification")
        return atlas

    if atlas.type == "cortical":
        geometry_df = geometry_df.copy()
        geometry_df["_hemi"] = geometry_df["label"].apply(_get_hemi_from_label)
        geometry_df["_group_key"] = geometry_df["_hemi"] + " " + geometry_df["view"]

        expanded_order = []
        for view in order:
            hemis_in_view = geometry_df.loc[geometry_df["view"] == view, "_hemi"].unique()
            for hemi in ["left", "right", ""]:
                if hemi in hemis_in_view:
                    expanded_order.append(f"{hemi} {view}")

        order_map = {key: idx for idx, key in enumerate(expanded_order)}
        fallback_order = len(expanded_order)
        geometry_df["_sort_order"] = geometry_df["_group_key"].map(
            lambda key: order_map.get(key, fallback_order)
        )
        geometry_df = geometry_df.sort_values("_sort_order")
        geometry_df = geometry_df.drop(columns=["_hemi", "_group_key", "_sort_order"])
    else:
        view_order_map = {view: idx for idx, view in enumerate(order)}
        geometry_df = geometry_df.copy()
        geometry_df["_sort_order"] = geometry_df["view"].map(view_order_map).fillna(len(order))
        geometry_df = geometry_df.sort_values("_sort_order")
        geometry_df = geometry_df.drop(columns=["_sort_order"])

    new_geometry = _reposition_views(geometry_df, atlas_type=atlas.type, gap=gap)
    new_data = _rebuild_atlas_data(atlas, new_geometry)
    return _clone_atlas(atlas, data=new_data)


def _filter_dataframe_by_labels(
    dataframe: pd.DataFrame | gpd.GeoDataFrame | None,
    labels_to_remove: set[str],
) -> pd.DataFrame | gpd.GeoDataFrame | None:
    """Filter dataframe to exclude specified labels."""
    if dataframe is None:
        return None
    return dataframe[~dataframe["label"].isin(labels_to_remove)].copy()


def _rebuild_data_without_labels(
    atlas: BrainAtlas,
    labels_to_remove: set[str],
    remove_sf: bool = True,
) -> AtlasData:
    """Rebuild atlas data container without specified labels."""
    from ggsegpy.data import CorticalData, SubcorticalData, TractData

    new_ggseg = atlas.data.ggseg
    if remove_sf:
        new_ggseg = _filter_dataframe_by_labels(new_ggseg, labels_to_remove)

    new_ggseg3d = _filter_dataframe_by_labels(atlas.data.ggseg3d, labels_to_remove)

    if isinstance(atlas.data, CorticalData):
        return CorticalData(ggseg=new_ggseg, ggseg3d=new_ggseg3d, mesh=atlas.data.mesh)
    if isinstance(atlas.data, SubcorticalData):
        return SubcorticalData(ggseg=new_ggseg, ggseg3d=new_ggseg3d)
    if isinstance(atlas.data, TractData):
        return TractData(ggseg=new_ggseg, ggseg3d=new_ggseg3d)

    raise TypeError(f"Unknown atlas data type: {type(atlas.data).__name__}")


def _rebuild_atlas_data(atlas: BrainAtlas, new_geometry: gpd.GeoDataFrame) -> AtlasData:
    """Rebuild atlas data container with new sf data."""
    from ggsegpy.data import CorticalData, SubcorticalData, TractData

    if isinstance(atlas.data, CorticalData):
        return CorticalData(
            ggseg=new_geometry, ggseg3d=atlas.data.ggseg3d, mesh=atlas.data.mesh
        )
    if isinstance(atlas.data, SubcorticalData):
        return SubcorticalData(ggseg=new_geometry, ggseg3d=atlas.data.ggseg3d)
    if isinstance(atlas.data, TractData):
        return TractData(ggseg=new_geometry, ggseg3d=atlas.data.ggseg3d)

    raise TypeError(f"Unknown atlas data type: {type(atlas.data).__name__}")


def _reposition_views(
    geometry_df: gpd.GeoDataFrame,
    atlas_type: str | None = None,
    gap: float = 0.15,
) -> gpd.GeoDataFrame:
    """Reposition view groups horizontally with gap spacing."""
    from shapely.affinity import translate

    geometry_df = geometry_df.copy()

    if atlas_type == "cortical":
        geometry_df["_group_key"] = (
            geometry_df["label"].apply(_get_hemi_from_label) + " " + geometry_df["view"]
        )
    else:
        geometry_df["_group_key"] = geometry_df["view"]

    group_names = geometry_df["_group_key"].unique()

    view_groups = []
    for group_name in group_names:
        mask = geometry_df["_group_key"] == group_name
        group_df = geometry_df[mask].copy()

        bounds = group_df.total_bounds
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2

        offset_x, offset_y = -center_x, -center_y
        group_df["geometry"] = group_df["geometry"].apply(
            lambda geom, dx=offset_x, dy=offset_y: translate(geom, dx, dy)
        )
        view_groups.append(group_df)

    group_bounds = []
    for group_df in view_groups:
        bounds = group_df.total_bounds
        group_bounds.append({
            "x_range": (bounds[0], bounds[2]),
            "y_range": (bounds[1], bounds[3]),
        })

    widths = [bounds["x_range"][1] - bounds["x_range"][0] for bounds in group_bounds]
    half_widths = [
        max(abs(bounds["x_range"][0]), abs(bounds["x_range"][1]))
        for bounds in group_bounds
    ]
    max_half_height = max(
        max(abs(bounds["y_range"][0]), abs(bounds["y_range"][1]))
        for bounds in group_bounds
    )
    gap_size = max(widths) * gap if widths else 0

    x_position = 0
    for idx, group_df in enumerate(view_groups):
        offset_x = x_position + half_widths[idx]
        offset_y = max_half_height
        group_df["geometry"] = group_df["geometry"].apply(
            lambda geom, dx=offset_x, dy=offset_y: translate(geom, dx, dy)
        )
        x_position += widths[idx] + gap_size

    result = pd.concat(view_groups, ignore_index=True)
    result = result.drop(columns=["_group_key"])

    return gpd.GeoDataFrame(result, geometry="geometry")
