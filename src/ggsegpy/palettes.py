from __future__ import annotations

from typing import Sequence

import numpy as np


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_palette(
    labels: Sequence[str],
    colormap: str = "viridis",
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(colormap)
        colors = [cmap(i / len(labels)) for i in range(len(labels))]
        return {
            label: rgb_to_hex(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            for label, c in zip(labels, colors)
        }
    except ImportError:
        hues = np.linspace(0, 360, len(labels), endpoint=False)
        return {label: f"hsl({int(h)}, 70%, 50%)" for label, h in zip(labels, hues)}


def scale_fill_brain(
    palette: dict[str, str],
    na_value: str = "grey",
) -> dict[str, str]:
    pal = palette.copy()
    pal["NA"] = na_value
    return pal
