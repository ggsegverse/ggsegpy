from __future__ import annotations

from typing import Literal

import geopandas as gpd
import numpy as np


def position_brain(
    gdf: gpd.GeoDataFrame,
    position: Literal["horizontal", "vertical", "stacked"] = "horizontal",
    hemi: list[str] | None = None,
    view: list[str] | None = None,
    spacing: float = 0.05,
) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    if hemi is not None:
        gdf = gdf[gdf["hemi"].isin(hemi)]

    if view is not None:
        gdf = gdf[gdf["view"].isin(view)]

    if "view" not in gdf.columns or "hemi" not in gdf.columns:
        return gdf

    views = gdf["view"].unique()
    hemis = gdf["hemi"].unique()

    bounds = gdf.total_bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    offset_x = width * (1 + spacing)
    offset_y = height * (1 + spacing)

    if position == "horizontal":
        gdf = _apply_horizontal_layout(gdf, hemis, views, offset_x)
    elif position == "vertical":
        gdf = _apply_vertical_layout(gdf, hemis, views, offset_y)
    elif position == "stacked":
        gdf = _apply_stacked_layout(gdf, hemis, views, offset_x, offset_y)

    return gdf


def _apply_horizontal_layout(
    gdf: gpd.GeoDataFrame,
    hemis: np.ndarray,
    views: np.ndarray,
    offset_x: float,
) -> gpd.GeoDataFrame:
    col_idx = 0
    for hemi in hemis:
        for view in views:
            mask = (gdf["hemi"] == hemi) & (gdf["view"] == view)
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].translate(
                xoff=col_idx * offset_x, yoff=0
            )
            col_idx += 1
    return gdf


def _apply_vertical_layout(
    gdf: gpd.GeoDataFrame,
    hemis: np.ndarray,
    views: np.ndarray,
    offset_y: float,
) -> gpd.GeoDataFrame:
    row_idx = 0
    for hemi in hemis:
        for view in views:
            mask = (gdf["hemi"] == hemi) & (gdf["view"] == view)
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].translate(
                xoff=0, yoff=-row_idx * offset_y
            )
            row_idx += 1
    return gdf


def _apply_stacked_layout(
    gdf: gpd.GeoDataFrame,
    hemis: np.ndarray,
    views: np.ndarray,
    offset_x: float,
    offset_y: float,
) -> gpd.GeoDataFrame:
    for row_idx, hemi in enumerate(hemis):
        for col_idx, view in enumerate(views):
            mask = (gdf["hemi"] == hemi) & (gdf["view"] == view)
            gdf.loc[mask, "geometry"] = gdf.loc[mask, "geometry"].translate(
                xoff=col_idx * offset_x, yoff=-row_idx * offset_y
            )
    return gdf


def get_view_order(views: list[str]) -> list[str]:
    view_priority = {
        "lateral": 0,
        "medial": 1,
        "superior": 2,
        "inferior": 3,
        "anterior": 4,
        "posterior": 5,
    }
    return sorted(views, key=lambda v: view_priority.get(v, 99))


def get_hemi_order(hemis: list[str]) -> list[str]:
    hemi_priority = {"left": 0, "right": 1, "midline": 2, "subcort": 3}
    return sorted(hemis, key=lambda h: hemi_priority.get(h, 99))
