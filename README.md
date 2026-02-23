

# ggsegpy

Brain atlas visualization in Python — port of the R
[ggseg](https://ggseg.github.io/ggseg/) ecosystem.

[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
[![PyPI](https://img.shields.io/pypi/v/ggsegpy.svg)](https://pypi.org/project/ggsegpy/)
[![Tests](https://github.com/ggsegverse/ggsegpy/actions/workflows/test.yml/badge.svg)](https://github.com/ggsegverse/ggsegpy/actions/workflows/test.yml)
[![Documentation](https://github.com/ggsegverse/ggsegpy/actions/workflows/docs.yml/badge.svg)](https://ggsegverse.github.io/ggsegpy/)

## Installation

``` bash
pip install git+https://github.com/ggsegverse/ggsegpy.git
```

## 2D Visualization

``` python
from plotnine import ggplot, aes
from ggsegpy import geom_brain, dk

ggplot() + geom_brain(atlas=dk())
```

<img src="https://raw.githubusercontent.com/ggsegverse/ggsegpy/main/README_files/figure-commonmark/cell-2-output-1.png"
width="525" height="375" />

**With custom data:**

``` python
import pandas as pd

data = pd.DataFrame({
    "region": ["precentral", "postcentral", "superior frontal", "inferior parietal"],
    "value": [0.9, 0.7, 0.5, 0.3]
})

ggplot(data) + geom_brain(atlas=dk(), mapping=aes(fill="value"))
```

<img src="https://raw.githubusercontent.com/ggsegverse/ggsegpy/main/README_files/figure-commonmark/cell-3-output-1.png"
width="525" height="375" />

### Subcortical atlas

``` python
from ggsegpy import aseg

ggplot() + geom_brain(atlas=aseg())
```

<img src="https://raw.githubusercontent.com/ggsegverse/ggsegpy/main/README_files/figure-commonmark/cell-4-output-1.png"
width="525" height="375" />

## 3D Visualization

``` python
from ggsegpy import ggseg3d, pan_camera, add_glassbrain

# Basic 3D plot
fig = ggseg3d(atlas=dk())
fig = pan_camera(fig, "left lateral")
fig
```

``` python
# With glass brain overlay
fig = ggseg3d(atlas=aseg())
fig = add_glassbrain(fig, opacity=0.1)
fig = pan_camera(fig, "left lateral")
fig
```

## Atlas Manipulation

Filter, rename, and reorganize atlas regions:

``` python
from ggsegpy import atlas_region_keep, atlas_view_keep

# Keep only frontal regions, lateral view
frontal = atlas_region_keep(dk(), "frontal")
frontal = atlas_view_keep(frontal, "lateral")

ggplot() + geom_brain(atlas=frontal)
```

<img src="https://raw.githubusercontent.com/ggsegverse/ggsegpy/main/README_files/figure-commonmark/cell-7-output-1.png"
width="525" height="375" />

``` python
from ggsegpy import atlas_region_rename, atlas_regions

# Shorten region names
renamed = atlas_region_rename(dk(), "superior", "sup.")
print([r for r in atlas_regions(renamed) if "sup." in r][:3])
```

    ['banks of sup. temporal sulcus', 'sup. frontal', 'sup. parietal']

## Available Atlases

### Bundled atlases

Three atlases ship with ggsegpy — no download required:

| Atlas       | Type         | Description                         |
|-------------|--------------|-------------------------------------|
| `dk()`      | Cortical     | Desikan-Killiany parcellation       |
| `aseg()`    | Subcortical  | FreeSurfer subcortical segmentation |
| `tracula()` | White matter | TRACULA tract atlas                 |

### Downloadable atlases

The [ggsegverse](https://github.com/ggsegverse) provides 19 additional atlases.
Fetch them by name:

``` python
from ggsegpy import fetch_atlas

destrieux = fetch_atlas("ggsegDestrieux")
schaefer = fetch_atlas("ggsegSchaefer")
```

To see what's available:

``` python
from ggsegpy import list_atlases

for name, info in list_atlases().items():
    if info["exported"]:
        print(f"{name}: {info['title']}")
```

Current atlases: ggsegAAL, ggsegAicha, ggsegArslan, ggsegBrainnetome, ggsegBrodmann, ggsegCampbell, ggsegDKT, ggsegDestrieux, ggsegEconomo, ggsegFlechsig, ggsegGlasser, ggsegGordon, ggsegHO, ggsegICBM, ggsegIca, ggsegKleist, ggsegPower, ggsegSchaefer, ggsegYeo2011.

### Cache management

Downloaded atlases live in `~/.cache/ggsegpy/` so you only download them once.

``` python
from ggsegpy import clear_cache

clear_cache("ggsegDestrieux")  # remove one atlas
clear_cache()                   # remove everything
```

If you need a fresh copy — maybe the upstream data changed — force a re-download:

``` python
atlas = fetch_atlas("ggsegDKT", force=True)
```

### Air-gapped environments

No internet on your cluster?
Grab the files manually and drop them in the cache directory.

**1. Download from a connected machine**

Visit [ggsegpy-data releases](https://github.com/ggsegverse/ggsegpy-data/releases) and download the parquet files you need.
Each atlas has 2-3 files:

- `{atlas}_core.parquet` — region metadata (label, hemisphere, color)
- `{atlas}_2d.parquet` — 2D polygon geometry
- `{atlas}_3d.parquet` — 3D vertex indices (if available)

**2. Place files in cache**

```
~/.cache/ggsegpy/
└── ggsegDestrieux/
    └── v2.0.2/
        ├── destrieux_core.parquet
        ├── destrieux_2d.parquet
        └── destrieux_3d.parquet
```

The version folder (e.g., `v2.0.2`) must match the version in the registry — check the release tag.

**3. Copy registry.json**

Download [registry.json](https://github.com/ggsegverse/ggsegpy-data/blob/main/registry.json) and place it at `~/.cache/ggsegpy/registry.json`.

**4. Use normally**

``` python
atlas = fetch_atlas("ggsegDestrieux")  # uses cached files, no network request
```

## Features

ggsegpy gives you 2D visualization via [plotnine](https://plotnine.org/) (ggplot2 syntax) and 3D interactive plots via [Plotly](https://plotly.com/python/).
You can filter, rename, and reorder atlas regions with the `atlas_*` functions, merge your data using `brain_join()`, and style plots with `theme_brain()`, `theme_darkbrain()`, or `theme_custombrain()`.
For subcortical context in 3D, add a glass brain overlay.

## Documentation

Full documentation at
[ggsegverse.github.io/ggsegpy](https://ggsegverse.github.io/ggsegpy/)

## License

MIT
