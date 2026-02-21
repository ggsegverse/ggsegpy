from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass
class HemiMesh:
    vertices: pd.DataFrame
    faces: pd.DataFrame


@dataclass
class SurfaceMesh:
    lh: HemiMesh
    rh: HemiMesh


@dataclass
class CorticalData:
    ggseg: gpd.GeoDataFrame
    ggseg3d: pd.DataFrame
    mesh: SurfaceMesh | None = None

    def __post_init__(self) -> None:
        required_ggseg_cols = {"label", "hemi", "region", "geometry"}
        if not required_ggseg_cols.issubset(self.ggseg.columns):
            missing = required_ggseg_cols - set(self.ggseg.columns)
            raise ValueError(f"ggseg missing required columns: {missing}")

        required_3d_cols = {"label", "vertex_indices"}
        if not required_3d_cols.issubset(self.ggseg3d.columns):
            missing = required_3d_cols - set(self.ggseg3d.columns)
            raise ValueError(f"ggseg3d missing required columns: {missing}")


@dataclass
class SubcorticalData:
    ggseg: gpd.GeoDataFrame | None
    ggseg3d: pd.DataFrame

    def __post_init__(self) -> None:
        required_3d_cols = {"label", "vertices", "faces"}
        if not required_3d_cols.issubset(self.ggseg3d.columns):
            missing = required_3d_cols - set(self.ggseg3d.columns)
            raise ValueError(f"ggseg3d missing required columns: {missing}")


@dataclass
class TractData:
    ggseg: gpd.GeoDataFrame | None
    ggseg3d: pd.DataFrame

    def __post_init__(self) -> None:
        required_3d_cols = {"label", "centerline"}
        if not required_3d_cols.issubset(self.ggseg3d.columns):
            missing = required_3d_cols - set(self.ggseg3d.columns)
            raise ValueError(f"ggseg3d missing required columns: {missing}")


AtlasData = CorticalData | SubcorticalData | TractData
