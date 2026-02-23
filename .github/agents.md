# Agent Context: ggsegpy

Python port of the R [ggseg](https://ggseg.github.io/ggseg/) brain atlas visualization ecosystem.

## What This Package Does

Renders brain atlases in 2D (plotnine/ggplot2 syntax) and 3D (plotly). Ships with three bundled atlases: Desikan-Killiany cortical parcellation (`dk`), FreeSurfer subcortical segmentation (`aseg`), and TRACULA white matter tracts (`tracula`).

## Architecture

```
src/ggsegpy/
├── geom_brain.py        # 2D: plotnine layer, defers creation until __radd__
├── plot3d.py            # 3D: plotly figures, singledispatch by atlas type
├── join.py              # Merges user data with atlas, handles faceting
├── atlas.py             # BrainAtlas class hierarchy
├── atlas_utils.py       # Manipulation: filter, rename, reorder
├── position_brain.py    # Spatial layout of views/hemispheres
├── themes.py            # plotnine themes
├── palettes.py          # Color scales
└── atlases/data/        # Parquet files (tracked in git)
```

## Three Patterns You Must Understand

### 1. Deferred Layer Creation

`geom_brain()` cannot access ggplot data at call time. It returns a `_BrainLayers` object that builds actual layers when added via `__radd__`:

```python
# In _BrainLayers:
def __radd__(self, gg):
    gg_data = getattr(gg, "data", None)  # Now we can access it
    layers = self._build_layers(gg_data)
    for layer in layers:
        layer.__radd__(gg)  # Modifies in-place, no return value
```

### 2. Atlas Type Dispatch

3D rendering differs by atlas type. Cortical uses vertex coloring on shared meshes; subcortical renders separate meshes per region:

```python
@singledispatch
def _add_atlas_surfaces(fig, atlas, ...): ...

@_add_atlas_surfaces.register(CorticalAtlas)
def _add_cortical_surfaces(fig, atlas, ...): ...

@_add_atlas_surfaces.register(SubcorticalAtlas)
def _add_subcortical_surfaces(fig, atlas, ...): ...
```

### 3. Automatic Facet Expansion

When user data has extra string columns (potential facet variables), `brain_join()` replicates the atlas for each unique combination. Without this, faceted plots show incomplete brains:

```python
# Detected: "group" is string, not a join column, not in atlas
# Result: atlas duplicated for group="A" and group="B"
data = pd.DataFrame({"region": [...], "group": ["A", "B"], "value": [...]})
joined = brain_join(data, atlas)  # Full brain for each group
```

## Things That Will Bite You

| Trap | Fix |
|------|-----|
| Using `"mono"` font | Use `"monospace"` (valid matplotlib family) |
| Region names like `"superiorfrontal"` | Atlas uses `"superior frontal"` (with space) |
| Checking `dtype == "object"` for strings | Use `pd.api.types.is_string_dtype()` |
| Expecting `__radd__` to return something | It modifies the ggplot in-place |
| Subcortical regions floating separately | Default position groups by hemi; use `position_brain(rows=None, cols="view")` |

## Join Column Priority

`brain_join()` auto-detects which column to merge on:

1. `label` — hemisphere prefix included (e.g., `"lh_precentral"`)
2. `region` — no prefix, matches both hemispheres (e.g., `"precentral"`)
3. `hemi` — hemisphere only (rarely useful alone)

Custom column names work if values match an atlas column: `brain_join(data, atlas, by="my_col")`.

## Commands

```bash
# Tests
pytest --timeout=60

# Lint
ruff check src/ && ruff format src/

# Render README (generates figures)
source .venv/bin/activate && quarto render README.qmd

# Preview docs
source .venv/bin/activate && quarto preview docs/
```

## File Locations

| What | Where |
|------|-------|
| Public API | `src/ggsegpy/__init__.py` |
| Atlas parquet data | `src/ggsegpy/atlases/data/*.parquet` |
| Tests | `tests/test_*.py` (mirror source files) |
| Docs source | `docs/` |
| README source | `README.qmd` → `README.md` + `README_files/` |

## Dependencies

- **plotnine**: 2D ggplot2 syntax
- **plotly**: 3D interactive figures
- **geopandas**: Polygon geometry
- **pyarrow**: Parquet I/O for bundled atlas data

## Status

Experimental. Not on PyPI. Install via `pip install git+https://github.com/ggsegverse/ggsegpy.git`.
