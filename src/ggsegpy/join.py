from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas


def brain_join(
    data: pd.DataFrame,
    atlas: BrainAtlas,
    by: str | list[str] | None = None,
) -> gpd.GeoDataFrame:
    """Merge user data with atlas geometry.

    Performs a left join, keeping all atlas regions with NaN for regions
    not in your data. Warns if data contains labels not found in the atlas.

    For faceted plots, extra columns in your data (beyond join and value columns)
    are treated as facet variables. The atlas is replicated for each unique
    combination of facet values, ensuring the full brain appears in every facet.

    Parameters
    ----------
    data
        DataFrame with values to map onto brain regions. Must contain a column
        that matches atlas labels.
    atlas
        Brain atlas to join with.
    by
        Column(s) to join on. If None, auto-detects from common columns
        (label, region, hemi). Can be a single column name or list.

    Returns
    -------
    gpd.GeoDataFrame
        Merged GeoDataFrame with atlas geometry and user data.

    Examples
    --------
    Join by label (includes hemisphere prefix):

    >>> from ggsegpy import dk, brain_join
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "label": ["lh_precentral", "rh_precentral"],
    ...     "thickness": [2.5, 2.1]
    ... })
    >>> merged = brain_join(data, dk())

    Join by region (applies to both hemispheres):

    >>> data = pd.DataFrame({
    ...     "region": ["precentral", "postcentral"],
    ...     "value": [0.9, 0.7]
    ... })
    >>> merged = brain_join(data, dk())

    Faceted data (atlas replicated per group):

    >>> data = pd.DataFrame({
    ...     "region": ["precentral", "precentral"],
    ...     "group": ["A", "B"],
    ...     "value": [0.9, 0.3]
    ... })
    >>> merged = brain_join(data, dk())  # Full brain for each group
    """
    sf = atlas.data.ggseg.copy()

    join_cols = _detect_join_columns(atlas, data, by)

    for col in join_cols:
        if col not in data.columns:
            raise ValueError(f"Join column '{col}' not found in data")

    primary_col = join_cols[0]
    atlas_col = primary_col

    if primary_col not in sf.columns:
        atlas_col = _infer_atlas_column(sf, data[primary_col])
        data = data.rename(columns={primary_col: atlas_col})
        join_cols = [atlas_col] + join_cols[1:]

    if atlas_col in sf.columns:
        atlas_values = set(sf[atlas_col].unique())
        data_values = set(data[atlas_col].unique())
        unmatched = data_values - atlas_values
        if unmatched:
            warnings.warn(
                f"Data contains values not in atlas '{atlas_col}' column: {unmatched}",
                UserWarning,
                stacklevel=2,
            )

    facet_cols = _detect_facet_columns(data, join_cols, sf.columns)

    if facet_cols:
        sf = _expand_atlas_for_facets(sf, data, facet_cols)
        join_cols = join_cols + facet_cols

    merged = sf.merge(data, on=join_cols, how="left")

    return merged


def _detect_facet_columns(
    data: pd.DataFrame,
    join_cols: list[str],
    atlas_cols: pd.Index,
) -> list[str]:
    atlas_col_set = set(atlas_cols)
    join_col_set = set(join_cols)
    facet_cols = []
    for col in data.columns:
        if col in join_col_set or col in atlas_col_set:
            continue
        if pd.api.types.is_string_dtype(data[col]) or isinstance(
            data[col].dtype, pd.CategoricalDtype
        ):
            facet_cols.append(col)
    return facet_cols


def _expand_atlas_for_facets(
    sf: gpd.GeoDataFrame,
    data: pd.DataFrame,
    facet_cols: list[str],
) -> gpd.GeoDataFrame:
    facet_combos = data[facet_cols].drop_duplicates()
    expanded_parts = []
    for _, row in facet_combos.iterrows():
        sf_copy = sf.copy()
        for col in facet_cols:
            sf_copy[col] = row[col]
        expanded_parts.append(sf_copy)
    return gpd.GeoDataFrame(pd.concat(expanded_parts, ignore_index=True))


def _infer_atlas_column(sf: gpd.GeoDataFrame, values: pd.Series) -> str:
    sample_values = set(values.dropna().head(10))
    for col in ["label", "region"]:
        if col in sf.columns:
            atlas_values = set(sf[col].unique())
            if sample_values & atlas_values:
                return col
    return "label"


def _detect_join_columns(
    atlas: BrainAtlas,
    data: pd.DataFrame,
    by: str | list[str] | None,
) -> list[str]:
    if by is not None:
        return [by] if isinstance(by, str) else by

    data_cols = set(data.columns)
    atlas_cols = {"label", "region", "hemi"}

    common = data_cols & atlas_cols

    if "label" in common:
        return ["label"]
    if "region" in common:
        if "hemi" in common:
            return ["region", "hemi"]
        return ["region"]
    if "hemi" in common:
        return ["hemi"]

    raise ValueError(
        f"Cannot auto-detect join columns. "
        f"Data columns: {data_cols}. "
        f"Expected one of: label, region, hemi. "
        f"Specify 'by' parameter explicitly."
    )
