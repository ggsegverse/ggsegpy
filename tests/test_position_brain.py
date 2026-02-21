from __future__ import annotations

import pytest
from plotnine import ggplot

from ggsegpy import dk, geom_brain, position_brain


class TestPositionBrain:
    def test_default_position(self):
        pos = position_brain()
        assert pos.rows == "hemi"
        assert pos.cols == "view"

    def test_custom_rows_cols(self):
        pos = position_brain(rows="view", cols="hemi")
        assert pos.rows == "view"
        assert pos.cols == "hemi"

    def test_single_column_layout(self):
        pos = position_brain(rows="hemi", cols=None)
        assert pos.rows == "hemi"
        assert pos.cols is None

    def test_single_row_layout(self):
        pos = position_brain(rows=None, cols="view")
        assert pos.rows is None
        assert pos.cols == "view"

    def test_side_filter(self):
        pos = position_brain(side="left")
        assert pos.side == "left"

    def test_apply_to_geodataframe(self):
        atlas = dk()
        gdf = atlas.data.ggseg.copy()
        pos = position_brain()
        result = pos.apply(gdf)
        assert len(result) == len(gdf)

    def test_apply_side_filter(self):
        atlas = dk()
        gdf = atlas.data.ggseg.copy()
        pos = position_brain(side="left")
        result = pos.apply(gdf)
        assert all(result["hemi"] == "left")


class TestGeomBrainWithPosition:
    def test_string_position_horizontal(self):
        p = ggplot() + geom_brain(atlas=dk(), position="horizontal")
        assert isinstance(p, ggplot)

    def test_string_position_vertical(self):
        p = ggplot() + geom_brain(atlas=dk(), position="vertical")
        assert isinstance(p, ggplot)

    def test_string_position_stacked(self):
        p = ggplot() + geom_brain(atlas=dk(), position="stacked")
        assert isinstance(p, ggplot)

    def test_position_brain_object(self):
        p = ggplot() + geom_brain(atlas=dk(), position=position_brain(rows="view", cols="hemi"))
        assert isinstance(p, ggplot)

    def test_position_with_side(self):
        p = ggplot() + geom_brain(atlas=dk(), position=position_brain(side="left"))
        assert isinstance(p, ggplot)
