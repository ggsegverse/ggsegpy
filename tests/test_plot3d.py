from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from ggsegpy import aseg, dk, ggseg3d, pan_camera, tracula


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
