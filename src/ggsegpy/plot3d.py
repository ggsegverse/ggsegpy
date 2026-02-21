from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ggsegpy.join import brain_join

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas


def ggseg3d(
    data: pd.DataFrame | None = None,
    atlas: BrainAtlas | None = None,
    hemisphere: str | list[str] | None = None,
    surface: str = "inflated",
    color: str | None = None,
    label_by: str = "region",
    text_by: str | None = None,
    palette: dict[str, str] | None = None,
    na_color: str = "darkgrey",
    na_alpha: float = 1.0,
    show_legend: bool = True,
    **kwargs: Any,
) -> go.Figure:
    """Create an interactive 3D brain atlas visualization.

    Parameters
    ----------
    data
        DataFrame with values to map onto brain regions. Must contain a column
        that matches atlas labels (typically 'label' or 'region').
    atlas
        Brain atlas to plot. Defaults to dk() if not specified.
    hemisphere
        Hemisphere(s) to show: 'left', 'right', or list of both.
    surface
        Surface type for cortical atlases: 'inflated', 'white', 'pial'.
    color
        Column name in data to use for coloring regions. If None, uses
        atlas palette colors.
    label_by
        Column for hover labels. Default is 'region'.
    text_by
        Additional column for hover text.
    palette
        Custom color palette as dict mapping labels to colors.
    na_color
        Color for regions with missing data. Default is 'darkgrey'.
    na_alpha
        Opacity for NA regions (0-1). Default is 1.0.
    show_legend
        Whether to show the legend.
    **kwargs
        Additional arguments.

    Returns
    -------
    go.Figure
        A plotly Figure object that can be displayed or further customized.

    Examples
    --------
    Basic atlas plot:

    >>> from ggsegpy import ggseg3d, dk
    >>> ggseg3d(atlas=dk())

    Plot with custom data:

    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     "label": ["lh_precentral", "rh_precentral"],
    ...     "value": [2.5, 2.1]
    ... })
    >>> ggseg3d(data=data, atlas=dk(), color="value")

    Set camera and background:

    >>> from ggsegpy import pan_camera, set_background
    >>> fig = ggseg3d(atlas=dk())
    >>> fig = pan_camera(fig, "left lateral")
    >>> fig = set_background(fig, "black")
    """
    if atlas is None:
        from ggsegpy.atlases import dk

        atlas = dk()

    if hemisphere is not None:
        hemis = [hemisphere] if isinstance(hemisphere, str) else hemisphere
        atlas = atlas.filter(hemi=hemis)

    merged_data = None
    if data is not None:
        merged_data = brain_join(data, atlas)

    fig = go.Figure()

    if atlas.type == "cortical":
        _add_cortical_surfaces(
            fig=fig,
            atlas=atlas,
            data=merged_data,
            color=color,
            label_by=label_by,
            text_by=text_by,
            palette=palette or atlas.palette,
            na_color=na_color,
            na_alpha=na_alpha,
        )
    elif atlas.type == "subcortical":
        _add_subcortical_meshes(
            fig=fig,
            atlas=atlas,
            data=merged_data,
            color=color,
            label_by=label_by,
            text_by=text_by,
            palette=palette or atlas.palette,
            na_color=na_color,
            na_alpha=na_alpha,
        )
    elif atlas.type == "tract":
        _add_tract_lines(
            fig=fig,
            atlas=atlas,
            data=merged_data,
            color=color,
            label_by=label_by,
            text_by=text_by,
            palette=palette or atlas.palette,
            na_color=na_color,
            na_alpha=na_alpha,
        )

    _configure_layout(fig, "white", show_legend)

    return fig


def _position_hemisphere(x: list[float], hemisphere: str) -> list[float]:
    """Shift hemisphere vertices so left and right don't overlap."""
    x_arr = np.array(x)
    x_range = x_arr.max() - x_arr.min()
    half_width = x_range / 2

    if hemisphere == "left":
        return (x_arr - half_width).tolist()
    elif hemisphere == "right":
        return (x_arr + half_width).tolist()
    return x


def _add_cortical_surfaces(
    fig: go.Figure,
    atlas: BrainAtlas,
    data: pd.DataFrame | None,
    color: str | None,
    label_by: str,
    text_by: str | None,
    palette: dict[str, str],
    na_color: str,
    na_alpha: float,
) -> None:
    ggseg3d_df = atlas.data.ggseg3d
    mesh = atlas.data.mesh
    core = atlas.core

    if mesh is None:
        return

    hemis_to_render = core["hemi"].unique()

    for hemi in hemis_to_render:
        hemi_short = "lh" if hemi == "left" else "rh"
        hemi_mesh = mesh.lh if hemi == "left" else mesh.rh

        vertices = hemi_mesh.vertices.copy()
        faces = hemi_mesh.faces

        n_vertices = len(vertices)
        vertex_colors = [na_color] * n_vertices

        hemi_core = core[core["hemi"] == hemi]
        for _, region_row in hemi_core.iterrows():
            label = region_row["label"]

            region_data = ggseg3d_df[ggseg3d_df["label"] == label]
            if region_data.empty:
                continue

            vertex_indices = region_data.iloc[0]["vertex_indices"]
            if vertex_indices is None or len(vertex_indices) == 0:
                continue

            region_color = _get_region_color(
                label=label,
                data=data,
                color_col=color,
                palette=palette,
                na_color=na_color,
            )

            for idx in vertex_indices:
                if 0 <= idx < n_vertices:
                    vertex_colors[idx] = region_color

        # Position hemisphere so left and right don't overlap
        x_positioned = _position_hemisphere(vertices["x"].tolist(), hemi)

        fig.add_trace(
            go.Mesh3d(
                x=x_positioned,
                y=vertices["y"].tolist(),
                z=vertices["z"].tolist(),
                i=faces["i"].tolist(),
                j=faces["j"].tolist(),
                k=faces["k"].tolist(),
                vertexcolor=vertex_colors,
                opacity=1.0,
                name=f"{hemi} hemisphere",
                hoverinfo="skip",
                showlegend=False,
            )
        )


def _add_subcortical_meshes(
    fig: go.Figure,
    atlas: BrainAtlas,
    data: pd.DataFrame | None,
    color: str | None,
    label_by: str,
    text_by: str | None,
    palette: dict[str, str],
    na_color: str,
    na_alpha: float,
) -> None:
    ggseg3d_df = atlas.data.ggseg3d
    core = atlas.core

    for _, region_row in core.iterrows():
        label = region_row["label"]

        mesh_data = ggseg3d_df[ggseg3d_df["label"] == label]
        if mesh_data.empty:
            continue

        vertices = mesh_data.iloc[0]["vertices"]
        faces = mesh_data.iloc[0]["faces"]

        if vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
            continue

        region_color = _get_region_color(
            label=label,
            data=data,
            color_col=color,
            palette=palette,
            na_color=na_color,
        )

        x = [v[0] for v in vertices]
        y = [v[1] for v in vertices]
        z = [v[2] for v in vertices]
        i = [f[0] for f in faces]
        j = [f[1] for f in faces]
        k = [f[2] for f in faces]

        # Position based on hemisphere
        hemi = region_row.get("hemi", "midline")
        x = _position_hemisphere(x, hemi)

        hover_label = region_row.get(label_by, label)
        hover_text = str(hover_label)
        if text_by and text_by in region_row:
            hover_text += f"<br>{region_row[text_by]}"

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y,
                z=z,
                i=i,
                j=j,
                k=k,
                color=region_color,
                opacity=1.0,
                name=label,
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=True,
            )
        )


def _add_tract_lines(
    fig: go.Figure,
    atlas: BrainAtlas,
    data: pd.DataFrame | None,
    color: str | None,
    label_by: str,
    text_by: str | None,
    palette: dict[str, str],
    na_color: str,
    na_alpha: float,
) -> None:
    ggseg3d_df = atlas.data.ggseg3d
    core = atlas.core

    for _, region_row in core.iterrows():
        label = region_row["label"]

        line_data = ggseg3d_df[ggseg3d_df["label"] == label]
        if line_data.empty:
            continue

        centerline = line_data.iloc[0]["centerline"]
        if centerline is None or len(centerline) == 0:
            continue

        region_color = _get_region_color(
            label=label,
            data=data,
            color_col=color,
            palette=palette,
            na_color=na_color,
        )

        x = [p[0] for p in centerline]
        y = [p[1] for p in centerline]
        z = [p[2] for p in centerline]

        # Position based on hemisphere
        hemi = region_row.get("hemi", "midline")
        x = _position_hemisphere(x, hemi)

        hover_label = region_row.get(label_by, label)
        hover_text = str(hover_label)
        if text_by and text_by in region_row:
            hover_text += f"<br>{region_row[text_by]}"

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(
                    color=region_color,
                    width=4,
                ),
                opacity=1.0,
                name=label,
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=True,
            )
        )


def _get_region_color(
    label: str,
    data: pd.DataFrame | None,
    color_col: str | None,
    palette: dict[str, str],
    na_color: str,
) -> str:
    if data is not None and color_col:
        row = data[data["label"] == label]
        if not row.empty:
            val = row[color_col].iloc[0]
            if pd.notna(val):
                if isinstance(val, str):
                    if val in palette:
                        return palette[val]
                    if val.startswith("#") or val.startswith("rgb"):
                        return val
                    return na_color
                if isinstance(val, (int, float)):
                    return _value_to_color(val, data[color_col])
                return na_color
            return na_color

    return palette.get(label, na_color)


def _value_to_color(
    value: float,
    series: pd.Series,
    low_color: tuple[int, int, int] = (65, 105, 225),
    high_color: tuple[int, int, int] = (220, 20, 60),
) -> str:
    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        t = 0.5
    else:
        t = (value - min_val) / (max_val - min_val)

    r = int(low_color[0] + t * (high_color[0] - low_color[0]))
    g = int(low_color[1] + t * (high_color[1] - low_color[1]))
    b = int(low_color[2] + t * (high_color[2] - low_color[2]))

    return f"rgb({r},{g},{b})"


def _configure_layout(
    fig: go.Figure,
    background: str,
    show_legend: bool,
) -> None:
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=background,
        ),
        paper_bgcolor=background,
        plot_bgcolor=background,
        showlegend=show_legend,
        legend=dict(
            itemsizing="constant",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )


def pan_camera(
    fig: go.Figure,
    camera: str = "left lateral",
) -> go.Figure:
    """Set camera position using R ggseg3d-style view names.

    Parameters
    ----------
    fig
        The plotly figure to modify.
    camera
        Camera view position. Available views:

        - 'left lateral', 'right lateral' - outside view of hemisphere
        - 'left medial', 'right medial' - inside (midline) view
        - 'left superior', 'right superior' - top view
        - 'left inferior', 'right inferior' - bottom view
        - 'left anterior', 'right anterior' - front view
        - 'left posterior', 'right posterior' - back view

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).

    Examples
    --------
    >>> fig = ggseg3d(atlas=dk())
    >>> fig = pan_camera(fig, "left lateral")
    >>> fig = pan_camera(fig, "right medial")
    """
    camera_positions = {
        "left lateral": dict(eye=dict(x=-2.0, y=0, z=0)),
        "right lateral": dict(eye=dict(x=2.0, y=0, z=0)),
        "left medial": dict(eye=dict(x=2.0, y=0, z=0)),
        "right medial": dict(eye=dict(x=-2.0, y=0, z=0)),
        "left superior": dict(eye=dict(x=-0.5, y=0, z=2.0)),
        "right superior": dict(eye=dict(x=0.5, y=0, z=2.0)),
        "left inferior": dict(eye=dict(x=-0.5, y=0, z=-2.0)),
        "right inferior": dict(eye=dict(x=0.5, y=0, z=-2.0)),
        "left anterior": dict(eye=dict(x=-0.5, y=2.0, z=0)),
        "right anterior": dict(eye=dict(x=0.5, y=2.0, z=0)),
        "left posterior": dict(eye=dict(x=-0.5, y=-2.0, z=0)),
        "right posterior": dict(eye=dict(x=0.5, y=-2.0, z=0)),
        "superior": dict(eye=dict(x=0, y=0, z=2.0)),
        "inferior": dict(eye=dict(x=0, y=0, z=-2.0)),
        "anterior": dict(eye=dict(x=0, y=2.0, z=0)),
        "posterior": dict(eye=dict(x=0, y=-2.0, z=0)),
    }

    view_key = camera.lower()
    cam = camera_positions.get(view_key, camera_positions["left lateral"])
    fig.update_layout(scene_camera=cam)

    return fig


def set_background(
    fig: go.Figure,
    colour: str = "#ffffff",
) -> go.Figure:
    """Set the background color of the 3D plot.

    Parameters
    ----------
    fig
        The plotly figure to modify.
    colour
        Background color as hex string or color name.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    fig.update_layout(
        scene=dict(bgcolor=colour),
        paper_bgcolor=colour,
        plot_bgcolor=colour,
    )
    return fig


def add_glassbrain(
    fig: go.Figure,
    hemisphere: str | list[str] | None = None,
    colour: str = "#CCCCCC",
    opacity: float = 0.3,
    surface: str = "inflated",
) -> go.Figure:
    """Add a transparent glass brain mesh for context.

    Parameters
    ----------
    fig
        The plotly figure to add glassbrain to.
    hemisphere
        Hemisphere(s) to add: 'left', 'right', or list of both.
        None adds both hemispheres.
    colour
        Color of the glass brain surface.
    opacity
        Transparency level (0-1). Default is 0.3.
    surface
        Surface type: 'inflated', 'white', 'pial'.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).

    Examples
    --------
    >>> fig = ggseg3d(atlas=aseg())
    >>> fig = add_glassbrain(fig, opacity=0.1)
    >>> fig = pan_camera(fig, "left lateral")
    """
    from ggsegpy.atlases import dk

    atlas = dk()
    mesh = atlas.data.mesh

    if mesh is None:
        return fig

    if hemisphere is None:
        hemis = ["left", "right"]
    elif isinstance(hemisphere, str):
        hemis = [hemisphere]
    else:
        hemis = hemisphere

    for hemi in hemis:
        hemi_mesh = mesh.lh if hemi == "left" else mesh.rh
        vertices = hemi_mesh.vertices
        faces = hemi_mesh.faces

        x = _position_hemisphere(vertices["x"].tolist(), hemi)

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=vertices["y"].tolist(),
                z=vertices["z"].tolist(),
                i=faces["i"].tolist(),
                j=faces["j"].tolist(),
                k=faces["k"].tolist(),
                color=colour,
                opacity=opacity,
                name=f"glassbrain_{hemi}",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


def set_legend(
    fig: go.Figure,
    show: bool = True,
) -> go.Figure:
    """Configure the legend visibility.

    Parameters
    ----------
    fig
        The plotly figure to modify.
    show
        Whether to show the legend.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    fig.update_layout(showlegend=show)
    return fig


def remove_legend(fig: go.Figure) -> go.Figure:
    """Remove the legend from the plot.

    Parameters
    ----------
    fig
        The plotly figure to modify.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    return set_legend(fig, show=False)


def set_hover(
    fig: go.Figure,
    show: bool = True,
    template: str | None = None,
) -> go.Figure:
    """Configure hover information.

    Args:
        fig: The plotly figure
        show: Whether to show hover info
        template: Custom hover template string
    """
    hoverinfo = "text" if show else "skip"

    for trace in fig.data:
        trace.hoverinfo = hoverinfo
        if template and show:
            trace.hovertemplate = template

    return fig
