from __future__ import annotations

from functools import partial, singledispatch
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ggsegpy.join import brain_join

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas


class BrainFigure(go.Figure):
    """Plotly Figure with pipe operator support for R-style chaining.

    Extends go.Figure to allow pipe-style syntax:

        ggseg3d(atlas=dk()) | pan_camera("left lateral") | set_background("black")

    This mirrors R ggseg3d's pipe syntax:

        ggseg3d(atlas=dk) |> pan_camera("left lateral") |> set_background("black")
    """

    def __or__(self, other):
        """Enable pipe operator: fig | func(args)."""
        if callable(other):
            return other(self)
        if isinstance(other, partial):
            return other(self)
        raise TypeError(f"Cannot pipe to {type(other)}")


@singledispatch
def add_brain_meshes(
    atlas,
    fig: go.Figure,
    data: pd.DataFrame | None,
    color: str | None,
    label_by: str,
    text_by: str | None,
    palette: dict[str, str],
    na_color: str,
    na_alpha: float,
) -> None:
    """Add brain meshes to a plotly figure.

    This is a generic function that dispatches based on atlas type,
    similar to R's S3 method dispatch (e.g., prepare_brain_meshes.cortical_atlas).

    Parameters
    ----------
    atlas
        Brain atlas object (CorticalAtlas, SubcorticalAtlas, or TractAtlas).
    fig
        Plotly figure to add meshes to.
    data
        Merged data with color values.
    color
        Column name for coloring.
    label_by
        Column for hover labels.
    text_by
        Additional column for hover text.
    palette
        Color palette dict.
    na_color
        Color for missing values.
    na_alpha
        Opacity for NA regions.
    """
    raise NotImplementedError(
        f"No 3D rendering method for atlas type: {type(atlas).__name__}"
    )


def _register_dispatch_methods():
    """Register singledispatch methods for each atlas type.

    Called at module load to register CorticalAtlas, SubcorticalAtlas,
    and TractAtlas handlers. This is done in a function to avoid
    circular imports.
    """
    from ggsegpy.atlas import CorticalAtlas, SubcorticalAtlas, TractAtlas

    @add_brain_meshes.register(CorticalAtlas)
    def _add_cortical(
        atlas: CorticalAtlas,
        fig: go.Figure,
        data: pd.DataFrame | None,
        color: str | None,
        label_by: str,
        text_by: str | None,
        palette: dict[str, str],
        na_color: str,
        na_alpha: float,
    ) -> None:
        """Dispatch method for cortical atlases - vertex coloring on shared mesh."""
        _add_cortical_surfaces(
            fig=fig,
            atlas=atlas,
            data=data,
            color=color,
            label_by=label_by,
            text_by=text_by,
            palette=palette,
            na_color=na_color,
            na_alpha=na_alpha,
        )

    @add_brain_meshes.register(SubcorticalAtlas)
    def _add_subcortical(
        atlas: SubcorticalAtlas,
        fig: go.Figure,
        data: pd.DataFrame | None,
        color: str | None,
        label_by: str,
        text_by: str | None,
        palette: dict[str, str],
        na_color: str,
        na_alpha: float,
    ) -> None:
        """Dispatch method for subcortical atlases - per-region meshes."""
        _add_subcortical_meshes(
            fig=fig,
            atlas=atlas,
            data=data,
            color=color,
            label_by=label_by,
            text_by=text_by,
            palette=palette,
            na_color=na_color,
            na_alpha=na_alpha,
        )

    @add_brain_meshes.register(TractAtlas)
    def _add_tract(
        atlas: TractAtlas,
        fig: go.Figure,
        data: pd.DataFrame | None,
        color: str | None,
        label_by: str,
        text_by: str | None,
        palette: dict[str, str],
        na_color: str,
        na_alpha: float,
    ) -> None:
        """Dispatch method for tract atlases - tube meshes along centerlines."""
        _add_tract_tubes(
            fig=fig,
            atlas=atlas,
            data=data,
            color=color,
            label_by=label_by,
            text_by=text_by,
            palette=palette,
            na_color=na_color,
            na_alpha=na_alpha,
        )


_register_dispatch_methods()


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
) -> BrainFigure:
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

    fig = BrainFigure()

    # Dispatch to appropriate renderer based on atlas type
    # Uses singledispatch for S3-style method dispatch
    add_brain_meshes(
        atlas,
        fig=fig,
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
        hemi_mesh = mesh.lh if hemi == "left" else mesh.rh

        vertices = hemi_mesh.vertices.copy()
        faces = hemi_mesh.faces

        n_vertices = len(vertices)
        vertex_colors = np.array([na_color] * n_vertices, dtype=object)

        hemi_core = core[core["hemi"] == hemi]
        ggseg3d_indexed = ggseg3d_df.set_index("label")

        for label in hemi_core["label"]:
            if label not in ggseg3d_indexed.index:
                continue

            vertex_indices = ggseg3d_indexed.loc[label, "vertex_indices"]
            if vertex_indices is None or len(vertex_indices) == 0:
                continue

            region_color = _get_region_color(
                label=label,
                data=data,
                color_col=color,
                palette=palette,
                na_color=na_color,
            )

            indices = np.asarray(vertex_indices)
            valid_mask = (indices >= 0) & (indices < n_vertices)
            vertex_colors[indices[valid_mask]] = region_color

        vertex_colors = vertex_colors.tolist()

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

        verts_arr = np.asarray(vertices)
        faces_arr = np.asarray(faces)
        x, y, z = verts_arr[:, 0], verts_arr[:, 1], verts_arr[:, 2]
        i, j, k = faces_arr[:, 0], faces_arr[:, 1], faces_arr[:, 2]

        # Position based on hemisphere
        hemi = region_row.get("hemi", "midline")
        x = _position_hemisphere(x.tolist(), hemi)

        hover_label = region_row.get(label_by, label)
        hover_text = str(hover_label)
        if text_by and text_by in region_row:
            hover_text += f"<br>{region_row[text_by]}"

        fig.add_trace(
            go.Mesh3d(
                x=x,
                y=y.tolist(),
                z=z.tolist(),
                i=i.tolist(),
                j=j.tolist(),
                k=k.tolist(),
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

        pts = np.asarray(centerline)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

        # Position based on hemisphere
        hemi = region_row.get("hemi", "midline")
        x = _position_hemisphere(x.tolist(), hemi)
        y, z = y.tolist(), z.tolist()

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
    fig: go.Figure | str | None = None,
    camera: str = "left lateral",
) -> go.Figure | partial:
    """Set camera position using R ggseg3d-style view names.

    Supports both direct calls and pipe syntax:

        # Direct call
        fig = pan_camera(fig, "left lateral")

        # Pipe syntax
        fig = ggseg3d() | pan_camera("left lateral")

    Parameters
    ----------
    fig
        The plotly figure to modify. Can be omitted for pipe syntax.
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

    >>> # Or using pipe syntax:
    >>> fig = ggseg3d() | pan_camera("left lateral")
    """
    if fig is None or isinstance(fig, str):
        if isinstance(fig, str):
            camera = fig
        return partial(pan_camera, camera=camera)

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
    fig: go.Figure | str | None = None,
    colour: str = "#ffffff",
) -> go.Figure | partial:
    """Set the background color of the 3D plot.

    Supports both direct calls and pipe syntax:

        fig = set_background(fig, "black")
        fig = ggseg3d() | set_background("black")

    Parameters
    ----------
    fig
        The plotly figure to modify. Can be omitted for pipe syntax.
    colour
        Background color as hex string or color name.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    if fig is None or isinstance(fig, str):
        if isinstance(fig, str):
            colour = fig
        return partial(set_background, colour=colour)

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
    fig: go.Figure | bool | None = None,
    show: bool = True,
) -> go.Figure | partial:
    """Configure the legend visibility.

    Supports pipe syntax: ggseg3d() | set_legend(False)

    Parameters
    ----------
    fig
        The plotly figure to modify. Can be omitted for pipe syntax.
    show
        Whether to show the legend.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    if fig is None or isinstance(fig, bool):
        if isinstance(fig, bool):
            show = fig
        return partial(set_legend, show=show)

    fig.update_layout(showlegend=show)
    return fig


def remove_legend(fig: go.Figure | None = None) -> go.Figure | partial:
    """Remove the legend from the plot.

    Supports pipe syntax: ggseg3d() | remove_legend()

    Parameters
    ----------
    fig
        The plotly figure to modify. Can be omitted for pipe syntax.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    if fig is None:
        return partial(remove_legend)
    return set_legend(fig, show=False)


def set_hover(
    fig: go.Figure,
    show: bool = True,
    template: str | None = None,
) -> go.Figure:
    """Configure hover information.

    Parameters
    ----------
    fig
        The plotly figure to modify.
    show
        Whether to show hover info.
    template
        Custom hover template string.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    hoverinfo = "text" if show else "skip"

    for trace in fig.data:
        trace.hoverinfo = hoverinfo
        if template and show:
            trace.hovertemplate = template

    return fig


def set_flat_shading(
    fig: go.Figure | bool | None = None,
    flat: bool = True,
) -> go.Figure | partial:
    """Set flat shading mode for mesh surfaces.

    Flat shading uses a single color per face rather than smooth
    interpolation across vertices. This gives a faceted appearance
    that can help distinguish region boundaries.

    Supports pipe syntax: ggseg3d() | set_flat_shading(True)

    Parameters
    ----------
    fig
        The plotly figure to modify. Can be omitted for pipe syntax.
    flat
        Whether to use flat shading.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    if fig is None or isinstance(fig, bool):
        if isinstance(fig, bool):
            flat = fig
        return partial(set_flat_shading, flat=flat)

    for trace in fig.data:
        if isinstance(trace, go.Mesh3d):
            trace.flatshading = flat

    return fig


def set_orthographic(
    fig: go.Figure | bool | None = None,
    orthographic: bool = True,
) -> go.Figure | partial:
    """Toggle orthographic projection mode.

    Orthographic projection removes perspective distortion,
    making parallel lines appear parallel. This is useful for
    anatomical views where accurate proportions matter.

    Supports pipe syntax: ggseg3d() | set_orthographic(True)

    Parameters
    ----------
    fig
        The plotly figure to modify. Can be omitted for pipe syntax.
    orthographic
        Whether to use orthographic projection. False uses perspective.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    if fig is None or isinstance(fig, bool):
        if isinstance(fig, bool):
            orthographic = fig
        return partial(set_orthographic, orthographic=orthographic)

    projection_type = "orthographic" if orthographic else "perspective"
    fig.update_layout(scene_camera=dict(projection=dict(type=projection_type)))
    return fig


def set_opacity(
    fig: go.Figure | float | None = None,
    opacity: float = 1.0,
    traces: str | list[str] | None = None,
) -> go.Figure | partial:
    """Set opacity for mesh traces.

    Supports pipe syntax: ggseg3d() | set_opacity(0.5)

    Parameters
    ----------
    fig
        The plotly figure to modify. Can be omitted for pipe syntax.
    opacity
        Opacity value between 0 (transparent) and 1 (opaque).
    traces
        Trace names to modify. If None, modifies all traces.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    if fig is None or isinstance(fig, (int, float)) and not isinstance(fig, go.Figure):
        if isinstance(fig, (int, float)):
            opacity = fig
        return partial(set_opacity, opacity=opacity, traces=traces)

    if traces is not None:
        trace_names = [traces] if isinstance(traces, str) else traces
    else:
        trace_names = None

    for trace in fig.data:
        if trace_names is None or trace.name in trace_names:
            trace.opacity = opacity

    return fig


def set_surface_color(
    fig: go.Figure,
    colour: str,
    traces: str | list[str] | None = None,
) -> go.Figure:
    """Set uniform color for mesh surfaces.

    Parameters
    ----------
    fig
        The plotly figure to modify.
    colour
        Color to apply as hex string or color name.
    traces
        Trace names to modify. If None, modifies all mesh traces.

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).
    """
    if traces is not None:
        trace_names = [traces] if isinstance(traces, str) else traces
    else:
        trace_names = None

    for trace in fig.data:
        if trace_names is None or trace.name in trace_names:
            if isinstance(trace, go.Mesh3d):
                trace.color = colour
                trace.vertexcolor = None

    return fig


def add_atlas(
    fig: go.Figure,
    atlas: BrainAtlas | None = None,
    data: pd.DataFrame | None = None,
    hemisphere: str | list[str] | None = None,
    color: str | None = None,
    palette: dict[str, str] | None = None,
    na_color: str = "darkgrey",
    opacity: float = 1.0,
) -> go.Figure:
    """Add an additional atlas to an existing figure.

    Useful for overlaying subcortical structures on cortical surfaces,
    or combining multiple atlases in a single visualization.

    Parameters
    ----------
    fig
        The plotly figure to add the atlas to.
    atlas
        Brain atlas to add.
    data
        DataFrame with values to map onto brain regions.
    hemisphere
        Hemisphere(s) to show: 'left', 'right', or list of both.
    color
        Column name in data to use for coloring regions.
    palette
        Custom color palette as dict mapping labels to colors.
    na_color
        Color for regions with missing data.
    opacity
        Opacity for the added atlas (0-1).

    Returns
    -------
    go.Figure
        The modified figure (allows chaining).

    Examples
    --------
    >>> from ggsegpy import ggseg3d, add_atlas, dk, aseg
    >>> fig = ggseg3d(atlas=dk())
    >>> fig = add_atlas(fig, atlas=aseg(), opacity=0.8)
    """
    if atlas is None:
        return fig

    from ggsegpy.join import brain_join

    if hemisphere is not None:
        hemis = [hemisphere] if isinstance(hemisphere, str) else hemisphere
        atlas = atlas.filter(hemi=hemis)

    merged_data = None
    if data is not None:
        merged_data = brain_join(data, atlas)

    use_palette = palette or atlas.palette

    if atlas.type == "cortical":
        _add_cortical_surfaces(
            fig=fig,
            atlas=atlas,
            data=merged_data,
            color=color,
            label_by="region",
            text_by=None,
            palette=use_palette,
            na_color=na_color,
            na_alpha=opacity,
        )
    elif atlas.type == "subcortical":
        _add_subcortical_meshes(
            fig=fig,
            atlas=atlas,
            data=merged_data,
            color=color,
            label_by="region",
            text_by=None,
            palette=use_palette,
            na_color=na_color,
            na_alpha=opacity,
        )
    elif atlas.type == "tract":
        _add_tract_tubes(
            fig=fig,
            atlas=atlas,
            data=merged_data,
            color=color,
            label_by="region",
            text_by=None,
            palette=use_palette,
            na_color=na_color,
            na_alpha=opacity,
        )

    for trace in fig.data:
        if hasattr(trace, "opacity") and trace.opacity is None:
            trace.opacity = opacity

    return fig


def _add_tract_tubes(
    fig: go.Figure,
    atlas: BrainAtlas,
    data: pd.DataFrame | None,
    color: str | None,
    label_by: str,
    text_by: str | None,
    palette: dict[str, str],
    na_color: str,
    na_alpha: float,
    radius: float = 1.0,
    n_sides: int = 8,
) -> None:
    """Add tract centerlines as tube meshes (like R ggseg3d)."""
    ggseg3d_df = atlas.data.ggseg3d
    core = atlas.core

    for _, region_row in core.iterrows():
        label = region_row["label"]

        line_data = ggseg3d_df[ggseg3d_df["label"] == label]
        if line_data.empty:
            continue

        centerline = line_data.iloc[0]["centerline"]
        if centerline is None or len(centerline) < 2:
            continue

        region_color = _get_region_color(
            label=label,
            data=data,
            color_col=color,
            palette=palette,
            na_color=na_color,
        )

        pts = np.asarray(centerline)

        hemi = region_row.get("hemi", "midline")
        if hemi == "left":
            pts[:, 0] = pts[:, 0] - np.ptp(pts[:, 0]) / 2
        elif hemi == "right":
            pts[:, 0] = pts[:, 0] + np.ptp(pts[:, 0]) / 2

        vertices, faces = _generate_tube_mesh(pts, radius=radius, n_sides=n_sides)

        hover_label = region_row.get(label_by, label)
        hover_text = str(hover_label)
        if text_by and text_by in region_row:
            hover_text += f"<br>{region_row[text_by]}"

        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0].tolist(),
                y=vertices[:, 1].tolist(),
                z=vertices[:, 2].tolist(),
                i=faces[:, 0].tolist(),
                j=faces[:, 1].tolist(),
                k=faces[:, 2].tolist(),
                color=region_color,
                opacity=na_alpha,
                name=label,
                hovertext=hover_text,
                hoverinfo="text",
                showlegend=True,
            )
        )


def _generate_tube_mesh(
    centerline: np.ndarray,
    radius: float = 1.0,
    n_sides: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a tube mesh around a centerline curve.

    Creates a cylindrical mesh that follows the path of the centerline,
    similar to R ggseg3d's tract tube rendering.

    Parameters
    ----------
    centerline
        Nx3 array of centerline points.
    radius
        Tube radius.
    n_sides
        Number of sides for the tube cross-section.

    Returns
    -------
    vertices
        Mx3 array of vertex positions.
    faces
        Fx3 array of triangle face indices.
    """
    n_points = len(centerline)
    if n_points < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=int)

    tangents = np.diff(centerline, axis=0)
    tangents = np.vstack([tangents, tangents[-1]])
    tangents = tangents / (np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-10)

    initial_normal = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tangents[0], initial_normal)) > 0.9:
        initial_normal = np.array([0.0, 1.0, 0.0])

    normals = np.zeros((n_points, 3))
    binormals = np.zeros((n_points, 3))

    normals[0] = np.cross(tangents[0], initial_normal)
    normals[0] /= np.linalg.norm(normals[0]) + 1e-10
    binormals[0] = np.cross(tangents[0], normals[0])

    for i in range(1, n_points):
        normals[i] = normals[i - 1] - np.dot(normals[i - 1], tangents[i]) * tangents[i]
        norm = np.linalg.norm(normals[i])
        if norm > 1e-10:
            normals[i] /= norm
        else:
            normals[i] = normals[i - 1]
        binormals[i] = np.cross(tangents[i], normals[i])

    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    vertices = np.zeros((n_points * n_sides, 3))
    for i in range(n_points):
        for j in range(n_sides):
            offset = radius * (cos_a[j] * normals[i] + sin_a[j] * binormals[i])
            vertices[i * n_sides + j] = centerline[i] + offset

    faces = []
    for i in range(n_points - 1):
        for j in range(n_sides):
            j_next = (j + 1) % n_sides

            v0 = i * n_sides + j
            v1 = i * n_sides + j_next
            v2 = (i + 1) * n_sides + j
            v3 = (i + 1) * n_sides + j_next

            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    return vertices, np.array(faces, dtype=int)
