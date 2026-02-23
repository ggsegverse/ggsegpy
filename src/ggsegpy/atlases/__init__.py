from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ggsegpy.atlas import CorticalAtlas, SubcorticalAtlas, TractAtlas
from ggsegpy.data import (
    BrainMeshes,
    CorticalData,
    HemiMesh,
    SubcorticalData,
    SurfaceMesh,
    TractData,
)

DATA_DIR = Path(__file__).parent / "data"


def dk() -> CorticalAtlas:
    core = _load_core("dk")
    ggseg = _load_ggseg("dk", core)
    ggseg3d = _load_cortical_3d("dk")
    mesh = _load_brain_meshes()
    palette = _extract_palette(core)

    return CorticalAtlas(
        atlas="dk",
        type="cortical",
        core=core,
        data=CorticalData(ggseg=ggseg, ggseg3d=ggseg3d, mesh=mesh),
        palette=palette,
    )


def aseg() -> SubcorticalAtlas:
    core = _load_core("aseg")
    ggseg = _load_ggseg("aseg", core)
    ggseg3d = _load_subcortical_3d("aseg")
    palette = _extract_palette(core)

    return SubcorticalAtlas(
        atlas="aseg",
        type="subcortical",
        core=core,
        data=SubcorticalData(ggseg=ggseg, ggseg3d=ggseg3d),
        palette=palette,
    )


def tracula() -> TractAtlas:
    core = _load_core("tracula")
    ggseg = _load_ggseg("tracula", core)
    ggseg3d = _load_tract_3d("tracula")
    palette = _extract_palette(core)

    return TractAtlas(
        atlas="tracula",
        type="tract",
        core=core,
        data=TractData(ggseg=ggseg, ggseg3d=ggseg3d),
        palette=palette,
    )


def _load_core(atlas: str) -> pd.DataFrame:
    path = DATA_DIR / f"{atlas}_core.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return _generate_placeholder_core(atlas)


def _load_ggseg(atlas: str, core: pd.DataFrame) -> gpd.GeoDataFrame:
    path = DATA_DIR / f"{atlas}_2d.parquet"
    if path.exists():
        from shapely import wkt

        df = pd.read_parquet(path)
        df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
        df = df.drop(columns=["geometry_wkt"])

        df = df.merge(core, on="label", how="left")

        hemi_missing = df["hemi"].isna()
        if hemi_missing.any():
            labels = df.loc[hemi_missing, "label"]
            inferred_hemi = pd.Series("midline", index=labels.index)
            inferred_hemi[labels.str.startswith("lh_")] = "left"
            inferred_hemi[labels.str.startswith("rh_")] = "right"
            df.loc[hemi_missing, "hemi"] = inferred_hemi

        region_missing = df["region"].isna()
        if region_missing.any():
            labels = df.loc[region_missing, "label"]
            has_prefix = labels.str.startswith(("lh_", "rh_"))
            inferred_region = labels.copy()
            inferred_region[has_prefix] = labels[has_prefix].str[3:]
            df.loc[region_missing, "region"] = inferred_region

        if "color" in df.columns:
            df["color"] = df["color"].fillna("#A9A9A9")

        return gpd.GeoDataFrame(df, geometry="geometry")
    return _generate_placeholder_ggseg(atlas)


def _load_brain_meshes() -> BrainMeshes | None:
    import numpy as np

    path = DATA_DIR / "brain_meshes.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)

    def make_hemi_mesh(row):
        verts = pd.DataFrame(
            {
                "x": np.array(row["vertices_x"]),
                "y": np.array(row["vertices_y"]),
                "z": np.array(row["vertices_z"]),
            }
        )
        faces = pd.DataFrame(
            {
                "i": np.array(row["faces_i"]),
                "j": np.array(row["faces_j"]),
                "k": np.array(row["faces_k"]),
            }
        )
        return HemiMesh(vertices=verts, faces=faces)

    surfaces = {}
    for surface_name in df["surface"].unique():
        surface_df = df[df["surface"] == surface_name]
        lh_row = surface_df[surface_df["hemi"] == "lh"].iloc[0]
        rh_row = surface_df[surface_df["hemi"] == "rh"].iloc[0]
        surfaces[surface_name] = SurfaceMesh(
            lh=make_hemi_mesh(lh_row),
            rh=make_hemi_mesh(rh_row),
        )

    return BrainMeshes(surfaces=surfaces)


def _load_cortical_3d(atlas: str) -> pd.DataFrame:
    path = DATA_DIR / f"{atlas}_3d.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "vertices" in df.columns and "vertex_indices" not in df.columns:
            df = df.rename(columns={"vertices": "vertex_indices"})
        return df
    return _generate_placeholder_vertex_indices(atlas)


def _load_subcortical_3d(atlas: str) -> pd.DataFrame:
    import numpy as np

    path = DATA_DIR / f"{atlas}_3d.parquet"
    if path.exists():
        df = pd.read_parquet(path)

        if "vertices_x" in df.columns:

            def build_vertices(row):
                vx = np.asarray(row["vertices_x"])
                vy = np.asarray(row["vertices_y"])
                vz = np.asarray(row["vertices_z"])
                return np.column_stack([vx, vy, vz]).tolist()

            def build_faces(row):
                fi = np.asarray(row["faces_i"]) - 1
                fj = np.asarray(row["faces_j"]) - 1
                fk = np.asarray(row["faces_k"]) - 1
                return np.column_stack([fi, fj, fk]).astype(int).tolist()

            df = pd.DataFrame(
                {
                    "label": df["label"],
                    "vertices": [build_vertices(row) for _, row in df.iterrows()],
                    "faces": [build_faces(row) for _, row in df.iterrows()],
                }
            )
        return df
    return _generate_placeholder_meshes(atlas)


def _load_tract_3d(atlas: str) -> pd.DataFrame:
    import numpy as np

    path = DATA_DIR / f"{atlas}_3d.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "points" in df.columns and "centerline" not in df.columns:
            df = df.rename(columns={"points": "centerline"})

        def reshape_centerline(arr):
            if arr is None or len(arr) == 0:
                return []
            arr = np.asarray(arr)
            n_points = len(arr) // 3
            if n_points == 0:
                return []
            return np.column_stack(
                [
                    arr[:n_points],
                    arr[n_points : 2 * n_points],
                    arr[2 * n_points : 3 * n_points],
                ]
            ).tolist()

        df["centerline"] = df["centerline"].apply(reshape_centerline)
        return df
    return _generate_placeholder_centerlines(atlas)


def _extract_palette(core: pd.DataFrame) -> dict[str, str]:
    if "color" in core.columns:
        colors = core["color"].fillna("#A9A9A9")
        return dict(zip(core["label"], colors))
    return {label: "#A9A9A9" for label in core["label"]}


def _generate_placeholder_core(atlas: str) -> pd.DataFrame:
    if atlas == "dk":
        regions = _dk_regions()
        rows = []
        for hemi in ["left", "right"]:
            prefix = "lh" if hemi == "left" else "rh"
            for region, color in regions.items():
                rows.append(
                    {
                        "label": f"{prefix}_{region}",
                        "hemi": hemi,
                        "region": region,
                        "color": color,
                    }
                )
        return pd.DataFrame(rows)
    elif atlas == "aseg":
        regions = _aseg_regions()
        return pd.DataFrame(
            [
                {"label": label, "hemi": hemi, "region": region, "color": color}
                for label, (hemi, region, color) in regions.items()
            ]
        )
    elif atlas == "tracula":
        regions = _tracula_regions()
        rows = []
        for hemi in ["left", "right"]:
            prefix = "lh" if hemi == "left" else "rh"
            for region, color in regions.items():
                rows.append(
                    {
                        "label": f"{prefix}_{region}",
                        "hemi": hemi,
                        "region": region,
                        "color": color,
                    }
                )
        return pd.DataFrame(rows)
    raise ValueError(f"Unknown atlas: {atlas}")


def _generate_placeholder_ggseg(atlas: str) -> gpd.GeoDataFrame:
    from shapely.geometry import box

    core = _generate_placeholder_core(atlas)
    rows = []

    for _, core_row in core.iterrows():
        label = core_row["label"]
        hemi = core_row["hemi"]
        region = core_row["region"]
        color = core_row["color"]

        for view_idx, view in enumerate(["lateral", "medial"]):
            geom = box(view_idx * 100, 0, view_idx * 100 + 10, 10)
            rows.append(
                {
                    "label": label,
                    "hemi": hemi,
                    "region": region,
                    "view": view,
                    "color": color,
                    "geometry": geom,
                }
            )

    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def _generate_placeholder_vertex_indices(atlas: str) -> pd.DataFrame:
    import numpy as np

    core = _generate_placeholder_core(atlas)
    rows = []
    for label in core["label"]:
        vertex_indices = list(np.random.randint(0, 10000, size=50))
        rows.append({"label": label, "vertex_indices": vertex_indices})
    return pd.DataFrame(rows)


def _generate_placeholder_meshes(atlas: str) -> pd.DataFrame:
    core = _generate_placeholder_core(atlas)
    rows = []
    for idx, label in enumerate(core["label"]):
        x_base = (idx % 5) * 20
        y_base = (idx // 5) * 20
        vertices = [
            [x_base, y_base, 0],
            [x_base + 10, y_base, 0],
            [x_base + 5, y_base + 10, 0],
            [x_base + 5, y_base + 5, 10],
        ]
        faces = [[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]]
        rows.append({"label": label, "vertices": vertices, "faces": faces})
    return pd.DataFrame(rows)


def _generate_placeholder_centerlines(atlas: str) -> pd.DataFrame:
    import numpy as np

    core = _generate_placeholder_core(atlas)
    rows = []
    for idx, label in enumerate(core["label"]):
        x_start = (idx % 4) * 30
        y_start = (idx // 4) * 30
        centerline = [
            [x_start + i * 3, y_start + np.sin(i / 3) * 10, i * 2] for i in range(20)
        ]
        rows.append({"label": label, "centerline": centerline})
    return pd.DataFrame(rows)


def _dk_regions() -> dict[str, str]:
    return {
        "bankssts": "#196428",
        "caudalanteriorcingulate": "#7D64A0",
        "caudalmiddlefrontal": "#641900",
        "cuneus": "#DC1464",
        "entorhinal": "#A08CB4",
        "frontalpole": "#640064",
        "fusiform": "#B4DC8C",
        "inferiorparietal": "#DC3CDC",
        "inferiortemporal": "#B42878",
        "insula": "#8C148C",
        "isthmuscingulate": "#8CDCDC",
        "lateraloccipital": "#0064C8",
        "lateralorbitofrontal": "#23A0DC",
        "lingual": "#DC781E",
        "medialorbitofrontal": "#C8A06E",
        "middletemporal": "#B400DC",
        "paracentral": "#B4FDC8",
        "parahippocampal": "#14DC3C",
        "parsopercularis": "#3CDC3C",
        "parsorbitalis": "#0096C8",
        "parstriangularis": "#508214",
        "pericalcarine": "#0A6414",
        "postcentral": "#14B4DC",
        "posteriorcingulate": "#00643C",
        "precentral": "#3C14DC",
        "precuneus": "#A0143C",
        "rostralanteriorcingulate": "#001432",
        "rostralmiddlefrontal": "#4B3264",
        "superiorfrontal": "#143264",
        "superiorparietal": "#14B432",
        "superiortemporal": "#64B4DC",
        "supramarginal": "#8CC814",
        "temporalpole": "#4B643C",
        "transversetemporal": "#A03264",
    }


def _aseg_regions() -> dict[str, tuple[str, str, str]]:
    return {
        "Left-Thalamus": ("left", "Thalamus", "#00760E"),
        "Left-Caudate": ("left", "Caudate", "#7D64A0"),
        "Left-Putamen": ("left", "Putamen", "#EC0DB0"),
        "Left-Pallidum": ("left", "Pallidum", "#0D30EC"),
        "Left-Hippocampus": ("left", "Hippocampus", "#DCE934"),
        "Left-Amygdala": ("left", "Amygdala", "#67FFFF"),
        "Left-Accumbens": ("left", "Accumbens", "#7FFF00"),
        "Right-Thalamus": ("right", "Thalamus", "#00760E"),
        "Right-Caudate": ("right", "Caudate", "#7D64A0"),
        "Right-Putamen": ("right", "Putamen", "#EC0DB0"),
        "Right-Pallidum": ("right", "Pallidum", "#0D30EC"),
        "Right-Hippocampus": ("right", "Hippocampus", "#DCE934"),
        "Right-Amygdala": ("right", "Amygdala", "#67FFFF"),
        "Right-Accumbens": ("right", "Accumbens", "#7FFF00"),
        "Brain-Stem": ("midline", "Brain-Stem", "#778899"),
    }


def _tracula_regions() -> dict[str, str]:
    return {
        "fmajor": "#FF0000",
        "fminor": "#FFA500",
        "atr": "#FFFF00",
        "cst": "#00FF00",
        "ilf": "#0000FF",
        "slf": "#8A2BE2",
        "unc": "#FF00FF",
        "ccg": "#00FFFF",
    }


__all__ = ["dk", "aseg", "tracula"]
