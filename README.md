# ggsegpy

Brain atlas visualization in Python — port of the R [ggseg](https://ggseg.github.io/ggseg/) ecosystem.

[![Tests](https://github.com/ggsegverse/ggsegpy/actions/workflows/test.yml/badge.svg)](https://github.com/ggsegverse/ggsegpy/actions/workflows/test.yml)
[![Documentation](https://github.com/ggsegverse/ggsegpy/actions/workflows/docs.yml/badge.svg)](https://ggsegverse.github.io/ggsegpy/)

## Installation

```bash
pip install ggsegpy
```

## Quick Start

```python
from plotnine import ggplot, aes
from ggsegpy import geom_brain, dk

# Basic 2D plot
ggplot() + geom_brain(atlas=dk())

# With data
import pandas as pd

data = pd.DataFrame({
    "region": ["precentral", "postcentral", "superiorfrontal"],
    "value": [0.5, 0.8, 0.3]
})
ggplot(data) + geom_brain(atlas=dk(), mapping=aes(fill="value"))
```

### 3D Visualization

```python
from ggsegpy import ggseg3d, pan_camera, add_glassbrain

fig = ggseg3d(atlas=dk())
fig = pan_camera(fig, "left lateral")
fig
```

## Available Atlases

| Atlas | Type | Description |
|-------|------|-------------|
| `dk()` | Cortical | Desikan-Killiany parcellation |
| `aseg()` | Subcortical | FreeSurfer subcortical segmentation |
| `tracula()` | White matter | TRACULA tract atlas |

## Features

- 2D visualization using [plotnine](https://plotnine.org/) (ggplot2 syntax)
- 3D interactive visualization using [Plotly](https://plotly.com/python/)
- Easy data merging with `brain_join()`
- Customizable themes: `theme_brain()`, `theme_darkbrain()`, `theme_custombrain()`
- Glass brain overlays for 3D subcortical context

## Documentation

Full documentation at [ggsegverse.github.io/ggsegpy](https://ggsegverse.github.io/ggsegpy/)

## License

MIT
