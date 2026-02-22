from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from ggsegpy import (
    add_atlas,
    add_glassbrain,
    aseg,
    dk,
    ggseg3d,
    pan_camera,
    set_background,
    set_flat_shading,
    set_legend,
    set_opacity,
    set_orthographic,
    set_surface_color,
    tracula,
)


class TestGgseg3d:
    def test_basic_plot(self):
        fig = ggseg3d(atlas=dk())
        assert isinstance(fig, go.Figure)

    def test_plot_with_data(self):
        data = pd.DataFrame({
            "label": ["lh_bankssts", "rh_bankssts"],
            "value": [0.5, 0.8],
        })
        fig = ggseg3d(data=data, atlas=dk(), color="value")
        assert isinstance(fig, go.Figure)

    def test_plot_default_atlas(self):
        fig = ggseg3d()
        assert isinstance(fig, go.Figure)

    def test_plot_filter_hemisphere(self):
        fig = ggseg3d(atlas=dk(), hemisphere="left")
        assert isinstance(fig, go.Figure)

    def test_plot_multiple_hemispheres(self):
        fig = ggseg3d(atlas=dk(), hemisphere=["left", "right"])
        assert isinstance(fig, go.Figure)

    def test_subcortical_atlas(self):
        fig = ggseg3d(atlas=aseg())
        assert isinstance(fig, go.Figure)

    def test_tract_atlas(self):
        fig = ggseg3d(atlas=tracula())
        assert isinstance(fig, go.Figure)

    def test_custom_palette(self):
        palette = {"lh_bankssts": "#FF0000", "rh_bankssts": "#00FF00"}
        fig = ggseg3d(atlas=dk(), palette=palette)
        assert isinstance(fig, go.Figure)


class TestPanCamera:
    def test_left_lateral(self):
        fig = ggseg3d(atlas=dk())
        fig = pan_camera(fig, "left lateral")
        assert fig.layout.scene.camera is not None

    def test_right_lateral(self):
        fig = ggseg3d(atlas=dk())
        fig = pan_camera(fig, "right lateral")
        assert fig.layout.scene.camera is not None

    def test_superior(self):
        fig = ggseg3d(atlas=dk())
        fig = pan_camera(fig, "superior")
        assert fig.layout.scene.camera is not None

    def test_left_medial(self):
        fig = ggseg3d(atlas=dk())
        fig = pan_camera(fig, "left medial")
        assert fig.layout.scene.camera is not None


class TestSetBackground:
    def test_set_white_background(self):
        fig = ggseg3d(atlas=dk())
        fig = set_background(fig, "#FFFFFF")
        assert fig.layout.scene.bgcolor == "#FFFFFF"

    def test_set_black_background(self):
        fig = ggseg3d(atlas=dk())
        fig = set_background(fig, "black")
        assert fig.layout.paper_bgcolor == "black"


class TestSetFlatShading:
    def test_enable_flat_shading(self):
        fig = ggseg3d(atlas=dk())
        fig = set_flat_shading(fig, flat=True)
        for trace in fig.data:
            if hasattr(trace, "flatshading"):
                assert trace.flatshading is True

    def test_disable_flat_shading(self):
        fig = ggseg3d(atlas=dk())
        fig = set_flat_shading(fig, flat=False)
        for trace in fig.data:
            if hasattr(trace, "flatshading"):
                assert trace.flatshading is False


class TestSetOrthographic:
    def test_enable_orthographic(self):
        fig = ggseg3d(atlas=dk())
        fig = set_orthographic(fig, orthographic=True)
        assert fig.layout.scene.camera.projection.type == "orthographic"

    def test_enable_perspective(self):
        fig = ggseg3d(atlas=dk())
        fig = set_orthographic(fig, orthographic=False)
        assert fig.layout.scene.camera.projection.type == "perspective"


class TestSetOpacity:
    def test_set_global_opacity(self):
        fig = ggseg3d(atlas=dk())
        fig = set_opacity(fig, opacity=0.5)
        for trace in fig.data:
            assert trace.opacity == 0.5

    def test_set_opacity_for_specific_trace(self):
        fig = ggseg3d(atlas=dk())
        fig = set_opacity(fig, opacity=0.3, traces="left hemisphere")
        for trace in fig.data:
            if trace.name == "left hemisphere":
                assert trace.opacity == 0.3


class TestSetLegend:
    def test_show_legend(self):
        fig = ggseg3d(atlas=dk())
        fig = set_legend(fig, show=True)
        assert fig.layout.showlegend is True

    def test_hide_legend(self):
        fig = ggseg3d(atlas=dk())
        fig = set_legend(fig, show=False)
        assert fig.layout.showlegend is False


class TestAddGlassbrain:
    def test_add_glassbrain_default(self):
        fig = ggseg3d(atlas=aseg())
        initial_traces = len(fig.data)
        fig = add_glassbrain(fig)
        assert len(fig.data) > initial_traces

    def test_add_glassbrain_single_hemisphere(self):
        fig = ggseg3d(atlas=aseg())
        initial_traces = len(fig.data)
        fig = add_glassbrain(fig, hemisphere="left")
        assert len(fig.data) == initial_traces + 1

    def test_add_glassbrain_with_opacity(self):
        fig = ggseg3d(atlas=aseg())
        fig = add_glassbrain(fig, opacity=0.1)
        glassbrain_traces = [t for t in fig.data if "glassbrain" in t.name]
        for trace in glassbrain_traces:
            assert trace.opacity == 0.1


class TestAddAtlas:
    def test_add_subcortical_to_cortical(self):
        fig = ggseg3d(atlas=dk())
        initial_traces = len(fig.data)
        fig = add_atlas(fig, atlas=aseg())
        assert len(fig.data) > initial_traces

    def test_add_atlas_with_opacity(self):
        fig = ggseg3d(atlas=dk())
        fig = add_atlas(fig, atlas=aseg(), opacity=0.7)
        assert isinstance(fig, go.Figure)


class TestChainedOperations:
    def test_chain_multiple_modifications(self):
        fig = ggseg3d(atlas=dk())
        fig = pan_camera(fig, "left lateral")
        fig = set_background(fig, "black")
        fig = set_flat_shading(fig, flat=True)
        fig = set_orthographic(fig, orthographic=True)
        assert isinstance(fig, go.Figure)
        assert fig.layout.scene.bgcolor == "black"
        assert fig.layout.scene.camera.projection.type == "orthographic"

    def test_method_chaining_returns_figure(self):
        fig = ggseg3d(atlas=dk())
        fig = pan_camera(fig, "left lateral")
        fig = set_background(fig, "#000000")
        fig = set_legend(fig, False)
        assert fig.layout.showlegend is False


class TestPipeSyntax:
    def test_pipe_pan_camera(self):
        fig = ggseg3d(atlas=dk()) | pan_camera("left lateral")
        assert isinstance(fig, go.Figure)
        assert fig.layout.scene.camera is not None

    def test_pipe_set_background(self):
        fig = ggseg3d(atlas=dk()) | set_background("black")
        assert fig.layout.scene.bgcolor == "black"

    def test_pipe_set_legend(self):
        fig = ggseg3d(atlas=dk()) | set_legend(False)
        assert fig.layout.showlegend is False

    def test_pipe_chain_multiple(self):
        fig = (
            ggseg3d(atlas=dk())
            | pan_camera("left lateral")
            | set_background("black")
            | set_flat_shading(True)
            | set_orthographic(True)
            | set_legend(False)
        )
        assert isinstance(fig, go.Figure)
        assert fig.layout.scene.bgcolor == "black"
        assert fig.layout.scene.camera.projection.type == "orthographic"
        assert fig.layout.showlegend is False

    def test_pipe_set_opacity(self):
        fig = ggseg3d(atlas=dk()) | set_opacity(0.5)
        for trace in fig.data:
            assert trace.opacity == 0.5
