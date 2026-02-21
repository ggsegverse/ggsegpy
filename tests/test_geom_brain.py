from __future__ import annotations

import pandas as pd
import pytest
from plotnine import ggplot

from ggsegpy import dk, ggseg, geom_brain


class TestGgseg:
    def test_basic_plot(self):
        p = ggseg(atlas=dk())
        assert isinstance(p, ggplot)

    def test_plot_with_data(self):
        data = pd.DataFrame({
            "label": ["lh_bankssts", "rh_bankssts"],
            "value": [0.5, 0.8],
        })
        p = ggseg(data=data, atlas=dk(), fill="value")
        assert isinstance(p, ggplot)

    def test_plot_default_atlas(self):
        p = ggseg()
        assert isinstance(p, ggplot)

    def test_plot_filter_hemi(self):
        p = ggseg(atlas=dk(), hemi="left")
        assert isinstance(p, ggplot)

    def test_plot_positions(self):
        for position in ["horizontal", "vertical", "stacked"]:
            p = ggseg(atlas=dk(), position=position)
            assert isinstance(p, ggplot)


class TestGeomBrain:
    def test_geom_brain_creation(self):
        gb = geom_brain(atlas=dk())
        assert gb.atlas is not None

    def test_geom_brain_callable(self):
        gb = geom_brain(atlas=dk())
        p = gb()
        assert isinstance(p, ggplot)

    def test_geom_brain_with_hemi(self):
        gb = geom_brain(atlas=dk(), hemi="left")
        assert gb.hemi == "left"

    def test_geom_brain_with_view(self):
        gb = geom_brain(atlas=dk(), view="lateral")
        assert gb.view == "lateral"
