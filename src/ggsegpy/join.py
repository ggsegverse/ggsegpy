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
    """
    sf = atlas.data.ggseg.copy()

    join_cols = _detect_join_columns(atlas, data, by)

    for col in join_cols:
        if col not in data.columns:
            raise ValueError(f"Join column '{col}' not found in data")

    atlas_labels = set(sf["label"].unique())
    data_labels = set(data[join_cols[0]].unique()) if len(join_cols) == 1 else None

    if data_labels:
        unmatched = data_labels - atlas_labels
        if unmatched:
            warnings.warn(
                f"Data contains labels not in atlas: {unmatched}",
                UserWarning,
                stacklevel=2,
            )

    if len(join_cols) == 1 and join_cols[0] != "label":
        data = data.rename(columns={join_cols[0]: "label"})
        join_cols = ["label"]

    merged = sf.merge(data, on=join_cols, how="left")

    return merged


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
