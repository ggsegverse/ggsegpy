from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlopen, urlretrieve

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas

REGISTRY_URL = (
    "https://api.github.com/repos/ggsegverse/ggsegpy-data/contents/registry.json"
)
CACHE_DIR = Path.home() / ".cache" / "ggsegpy"


def list_atlases(refresh: bool = False) -> dict[str, dict]:
    """List available atlases from the ggsegpy-data registry."""
    registry = _fetch_registry(refresh=refresh)
    return {
        name: {
            "version": info["version"],
            "title": info.get("title", name),
            "description": info.get("description", ""),
            "exported": info.get("exported", False),
            "n_files": len(info.get("files", [])),
        }
        for name, info in registry.items()
    }


def fetch_atlas(name: str, force: bool = False) -> BrainAtlas:
    """Fetch an atlas from ggsegpy-data and return as a BrainAtlas object.

    Args:
        name: Atlas package name (e.g., 'ggsegDKT', 'ggsegSchaefer')
        force: Re-download even if cached

    Returns:
        BrainAtlas object with core, 2D, and 3D data
    """
    registry = _fetch_registry()

    if name not in registry:
        available = [k for k, v in registry.items() if v.get("exported")]
        raise ValueError(
            f"Atlas '{name}' not found. Available: {', '.join(sorted(available))}"
        )

    info = registry[name]
    if not info.get("exported"):
        raise ValueError(f"Atlas '{name}' has not been exported yet")

    atlas_dir = CACHE_DIR / name / f"v{info['version']}"
    atlas_dir.mkdir(parents=True, exist_ok=True)

    for file_info in info.get("files", []):
        file_path = atlas_dir / file_info["name"]
        if (
            force
            or not file_path.exists()
            or not _verify_sha256(file_path, file_info.get("sha256"))
        ):
            _download_file(file_info["url"], file_path, file_info.get("sha256"))

    return _load_atlas_from_cache(name, atlas_dir, info)


def _fetch_registry(refresh: bool = False) -> dict:
    cache_path = CACHE_DIR / "registry.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not refresh and cache_path.exists():
        age_hours = (
            pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")
        ).total_seconds() / 3600
        if age_hours < 24:
            with open(cache_path) as f:
                return json.load(f)

    try:
        import base64

        with urlopen(REGISTRY_URL, timeout=10) as response:
            api_response = json.loads(response.read().decode())
            content = base64.b64decode(api_response["content"]).decode()
            data = json.loads(content)
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data
    except Exception:
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        raise


def _download_file(url: str, path: Path, expected_sha256: str | None = None):
    print(f"Downloading {path.name}...")
    urlretrieve(url, path)

    if expected_sha256 and not _verify_sha256(path, expected_sha256):
        path.unlink()
        raise ValueError(f"SHA256 mismatch for {path.name}")


def _verify_sha256(path: Path, expected: str | None) -> bool:
    if not expected or not path.exists():
        return False
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest() == expected


def _load_atlas_from_cache(name: str, atlas_dir: Path, info: dict) -> BrainAtlas:
    from ggsegpy.atlas import CorticalAtlas, SubcorticalAtlas
    from ggsegpy.data import CorticalData, SubcorticalData

    files = {f["name"]: atlas_dir / f["name"] for f in info.get("files", [])}

    atlas_type = "cortical"
    meta_files = [f for f in files if f.endswith("_meta.json")]
    if meta_files:
        with open(files[meta_files[0]]) as f:
            meta = json.load(f)
            atlas_type = meta.get("type", "cortical")

    core_files = [f for f in files if "_core.parquet" in f]

    atlases = {}
    for core_file in core_files:
        atlas_name = core_file.replace("_core.parquet", "")
        sf_file = f"{atlas_name}_2d.parquet"
        v3d_file = f"{atlas_name}_3d.parquet"

        core = pd.read_parquet(files[core_file])
        palette = (
            dict(zip(core["label"], core["color"])) if "color" in core.columns else {}
        )

        ggseg = None
        if sf_file in files:
            ggseg = _load_2d_data(files[sf_file], core)

        ggseg3d = None
        if v3d_file in files:
            ggseg3d = pd.read_parquet(files[v3d_file])
            if "vertices_json" in ggseg3d.columns:
                ggseg3d["vertex_indices"] = ggseg3d["vertices_json"].apply(json.loads)
                ggseg3d = ggseg3d.drop(columns=["vertices_json"])

        if atlas_type == "cortical":
            atlases[atlas_name] = CorticalAtlas(
                atlas=atlas_name,
                type="cortical",
                core=core[["hemi", "region", "label"]],
                data=CorticalData(ggseg=ggseg, ggseg3d=ggseg3d, mesh=None),
                palette=palette,
            )
        else:
            atlases[atlas_name] = SubcorticalAtlas(
                atlas=atlas_name,
                type="subcortical",
                core=core[["hemi", "region", "label"]],
                data=SubcorticalData(ggseg=ggseg, ggseg3d=ggseg3d),
                palette=palette,
            )

    if len(atlases) == 1:
        return list(atlases.values())[0]
    return atlases


def _load_2d_data(path: Path, core: pd.DataFrame) -> gpd.GeoDataFrame:
    from shapely import wkt

    df = pd.read_parquet(path)
    df["geometry"] = df["geometry_wkt"].apply(wkt.loads)
    df = df.drop(columns=["geometry_wkt"])
    df = df.merge(core, on="label", how="left")

    hemi_missing = df["hemi"].isna()
    if hemi_missing.any():
        labels = df.loc[hemi_missing, "label"]
        inferred = pd.Series("midline", index=labels.index)
        inferred[labels.str.startswith("lh_")] = "left"
        inferred[labels.str.startswith("rh_")] = "right"
        df.loc[hemi_missing, "hemi"] = inferred

    region_missing = df["region"].isna()
    if region_missing.any():
        labels = df.loc[region_missing, "label"]
        has_prefix = labels.str.startswith(("lh_", "rh_"))
        inferred = labels.copy()
        inferred[has_prefix] = labels[has_prefix].str[3:]
        df.loc[region_missing, "region"] = inferred

    if "color" in df.columns:
        df["color"] = df["color"].fillna("#A9A9A9")

    return gpd.GeoDataFrame(df, geometry="geometry")


def clear_cache(name: str | None = None):
    """Clear cached atlas data.

    Args:
        name: Specific atlas to clear, or None to clear all
    """
    if name:
        atlas_dir = CACHE_DIR / name
        if atlas_dir.exists():
            import shutil

            shutil.rmtree(atlas_dir)
    else:
        if CACHE_DIR.exists():
            import shutil

            shutil.rmtree(CACHE_DIR)


__all__ = ["list_atlases", "fetch_atlas", "clear_cache"]
