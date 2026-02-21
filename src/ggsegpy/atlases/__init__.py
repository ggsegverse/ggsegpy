from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from ggsegpy.atlas import CorticalAtlas, SubcorticalAtlas, TractAtlas
from ggsegpy.data import (
    CorticalData,
    HemiMesh,
    SubcorticalData,
    SurfaceMesh,
    TractData,
)

DATA_DIR = Path(__file__).parent / "data"


def dk() -> CorticalAtlas:
    ggseg = _load_or_placeholder_ggseg("dk")
    ggseg3d = _load_or_placeholder_cortical_3d("dk")
    mesh = _load_fsaverage5()
    palette = _extract_palette(ggseg)
    core = _extract_core(ggseg)

    return CorticalAtlas(
        atlas="dk",
        type="cortical",
        core=core,
        data=CorticalData(ggseg=ggseg, ggseg3d=ggseg3d, mesh=mesh),
        palette=palette,
    )


def aseg() -> SubcorticalAtlas:
    ggseg = _load_or_placeholder_ggseg("aseg")
    ggseg3d = _load_or_placeholder_subcortical_3d("aseg")
    palette = _extract_palette(ggseg)
    core = _extract_core(ggseg)

    return SubcorticalAtlas(
        atlas="aseg",
        type="subcortical",
        core=core,
        data=SubcorticalData(ggseg=ggseg, ggseg3d=ggseg3d),
        palette=palette,
    )


def tracula() -> TractAtlas:
    ggseg = _load_or_placeholder_ggseg("tracula")
    ggseg3d = _load_or_placeholder_tract_3d("tracula")
    palette = _extract_palette(ggseg)
    core = _extract_core(ggseg)

    return TractAtlas(
        atlas="tracula",
        type="tract",
        core=core,
        data=TractData(ggseg=ggseg, ggseg3d=ggseg3d),
        palette=palette,
    )


def _load_fsaverage5() -> SurfaceMesh | None:
    import numpy as np

    path = DATA_DIR / "fsaverage5.parquet"
    if path.exists():
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

        lh_row = df[df["hemi"] == "lh"].iloc[0]
        rh_row = df[df["hemi"] == "rh"].iloc[0]

        return SurfaceMesh(
            lh=make_hemi_mesh(lh_row),
            rh=make_hemi_mesh(rh_row),
        )
    return None


def _load_or_placeholder_ggseg(atlas: str) -> gpd.GeoDataFrame:
    path = DATA_DIR / f"{atlas}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "geometry_wkt" in df.columns:
            from shapely import wkt

            df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
            df = df.drop(columns=["geometry_wkt"])

        # Fill missing hemi values from label prefix for context regions
        def infer_hemi(row):
            if pd.notna(row["hemi"]):
                return row["hemi"]
            label = row["label"]
            if label.startswith("lh_"):
                return "left"
            elif label.startswith("rh_"):
                return "right"
            return "midline"

        df["hemi"] = df.apply(infer_hemi, axis=1)

        # Fill missing region from label
        def infer_region(row):
            if pd.notna(row["region"]):
                return row["region"]
            label = row["label"]
            if label.startswith(("lh_", "rh_")):
                return label[3:]
            return label

        df["region"] = df.apply(infer_region, axis=1)

        # Fill missing colors with grey for context regions
        if "color" in df.columns:
            df["color"] = df["color"].fillna("#A9A9A9")

        return gpd.GeoDataFrame(df, geometry="geometry")
    return _generate_placeholder_ggseg(atlas)


def _load_or_placeholder_cortical_3d(atlas: str) -> pd.DataFrame:
    path = DATA_DIR / f"{atlas}_3d.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "vertices" in df.columns and "vertex_indices" not in df.columns:
            df = df.rename(columns={"vertices": "vertex_indices"})
        return df
    return _generate_placeholder_vertex_indices(atlas)


def _load_or_placeholder_subcortical_3d(atlas: str) -> pd.DataFrame:
    import numpy as np

    path = DATA_DIR / f"{atlas}_3d.parquet"
    if path.exists():
        df = pd.read_parquet(path)

        if "vertices_x" in df.columns:
            vertices_list = []
            faces_list = []
            for _, row in df.iterrows():
                vx = np.array(row["vertices_x"])
                vy = np.array(row["vertices_y"])
                vz = np.array(row["vertices_z"])
                vertices = [
                    [float(vx[i]), float(vy[i]), float(vz[i])] for i in range(len(vx))
                ]
                vertices_list.append(vertices)

                fi = np.array(row["faces_i"])
                fj = np.array(row["faces_j"])
                fk = np.array(row["faces_k"])
                # Convert from R's 1-indexed to Python's 0-indexed
                faces = [
                    [int(fi[i]) - 1, int(fj[i]) - 1, int(fk[i]) - 1]
                    for i in range(len(fi))
                ]
                faces_list.append(faces)

            df = pd.DataFrame(
                {
                    "label": df["label"],
                    "vertices": vertices_list,
                    "faces": faces_list,
                }
            )
        return df
    return _generate_placeholder_meshes(atlas)


def _load_or_placeholder_tract_3d(atlas: str) -> pd.DataFrame:
    import numpy as np

    path = DATA_DIR / f"{atlas}_3d.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "points" in df.columns and "centerline" not in df.columns:
            df = df.rename(columns={"points": "centerline"})

        def reshape_centerline(arr):
            if arr is None or len(arr) == 0:
                return []
            arr = np.array(arr)
            n_points = len(arr) // 3
            if n_points == 0:
                return []
            x = arr[:n_points]
            y = arr[n_points : 2 * n_points]
            z = arr[2 * n_points : 3 * n_points]
            return [[float(x[i]), float(y[i]), float(z[i])] for i in range(n_points)]

        df["centerline"] = df["centerline"].apply(reshape_centerline)
        return df
    return _generate_placeholder_centerlines(atlas)


def _extract_palette(ggseg: gpd.GeoDataFrame) -> dict[str, str]:
    if "color" in ggseg.columns:
        palette = {}
        for label, color in zip(ggseg["label"], ggseg["color"]):
            if pd.isna(color):
                # Default grey for context regions (medial wall, unknown)
                palette[label] = "#A9A9A9"
            else:
                palette[label] = color
        return palette
    return _get_placeholder_palette(ggseg)


def _extract_core(ggseg: gpd.GeoDataFrame) -> pd.DataFrame:
    core_cols = ["hemi", "region", "label"]
    return ggseg[core_cols].drop_duplicates(subset=["label"]).reset_index(drop=True)


def _get_placeholder_palette(ggseg: gpd.GeoDataFrame) -> dict[str, str]:
    labels = ggseg["label"].unique()
    import numpy as np

    hues = np.linspace(0, 360, len(labels), endpoint=False)
    return {label: f"hsl({int(h)}, 70%, 50%)" for label, h in zip(labels, hues)}


def _generate_placeholder_ggseg(atlas: str) -> gpd.GeoDataFrame:
    from shapely.geometry import box

    if atlas == "dk":
        regions = _dk_regions()
        hemis = ["left", "right"]
    elif atlas == "aseg":
        regions = _aseg_regions()
        hemis = None
    elif atlas == "tracula":
        regions = _tracula_regions()
        hemis = ["left", "right"]
    else:
        raise ValueError(f"Unknown atlas: {atlas}")

    rows = []

    if hemis:
        region_list = list(regions.items())
        cols = 7

        for hemi_idx, hemi in enumerate(hemis):
            hemi_prefix = "lh" if hemi == "left" else "rh"
            for reg_idx, (region, color) in enumerate(region_list):
                label = f"{hemi_prefix}_{region}"
                col = reg_idx % cols
                row = reg_idx // cols

                for view_idx, view in enumerate(["lateral", "medial"]):
                    x = col * 15 + view_idx * (cols * 15 + 20)
                    y = row * 15 + hemi_idx * 80

                    geom = box(x, y, x + 12, y + 12)
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
    else:
        region_list = list(regions.items())
        cols = 5
        for reg_idx, (label, (hemi, region, color)) in enumerate(region_list):
            col = reg_idx % cols
            row = reg_idx // cols

            for view_idx, view in enumerate(["lateral", "medial"]):
                x = col * 15 + view_idx * (cols * 15 + 20)
                y = row * 15

                geom = box(x, y, x + 12, y + 12)
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

    if atlas == "dk":
        regions = _dk_regions()
        hemis = ["left", "right"]
    else:
        raise ValueError(f"No cortical placeholder for: {atlas}")

    rows = []
    for hemi in hemis:
        hemi_prefix = "lh" if hemi == "left" else "rh"
        for region in regions:
            label = f"{hemi_prefix}_{region}"
            vertex_indices = list(np.random.randint(0, 10000, size=50))
            rows.append({"label": label, "vertex_indices": vertex_indices})

    return pd.DataFrame(rows)


def _generate_placeholder_meshes(atlas: str) -> pd.DataFrame:
    if atlas == "aseg":
        regions = _aseg_regions()
    else:
        raise ValueError(f"No subcortical placeholder for: {atlas}")

    rows = []
    for idx, label in enumerate(regions.keys()):
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

    if atlas == "tracula":
        regions = _tracula_regions()
        hemis = ["left", "right"]
    else:
        raise ValueError(f"No tract placeholder for: {atlas}")

    rows = []
    idx = 0
    for hemi in hemis:
        hemi_prefix = "lh" if hemi == "left" else "rh"
        for region in regions:
            label = f"{hemi_prefix}_{region}"
            x_start = (idx % 4) * 30
            y_start = (idx // 4) * 30
            centerline = [
                [x_start + i * 3, y_start + np.sin(i / 3) * 10, i * 2]
                for i in range(20)
            ]
            rows.append({"label": label, "centerline": centerline})
            idx += 1

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
