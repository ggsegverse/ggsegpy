from __future__ import annotations

import pandas as pd
import pytest
from plotnine import aes, ggplot

from ggsegpy import dk, geom_brain


class TestGeomBrain:
    def test_basic_plot(self):
        p = ggplot() + geom_brain(atlas=dk())
        assert isinstance(p, ggplot)

    def test_plot_with_data(self):
        data = pd.DataFrame({
            "label": ["lh_bankssts", "rh_bankssts"],
            "value": [0.5, 0.8],
        })
        p = ggplot(data) + geom_brain(atlas=dk(), mapping=aes(fill="value"))
        assert isinstance(p, ggplot)

    def test_plot_default_atlas(self):
        p = ggplot() + geom_brain()
        assert isinstance(p, ggplot)

    def test_plot_filter_hemi(self):
        p = ggplot() + geom_brain(atlas=dk(), hemi="left")
        assert isinstance(p, ggplot)

    def test_plot_filter_view(self):
        p = ggplot() + geom_brain(atlas=dk(), view="lateral")
        assert isinstance(p, ggplot)

    def test_geom_brain_returns_layers(self):
        gb = geom_brain(atlas=dk(), hemi="left", view="lateral")
        assert hasattr(gb, "layers")
        assert len(gb.layers) > 0
