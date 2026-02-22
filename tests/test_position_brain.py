from __future__ import annotations

import numpy as np
import pytest

from ggsegpy import dk, position_brain


def get_panel_bounds(gdf, hemi, view):
    """Get bounds for a specific panel."""
    mask = np.ones(len(gdf), dtype=bool)
    if hemi is not None:
        mask &= gdf["hemi"] == hemi
    if view is not None:
        mask &= gdf["view"] == view
    return gdf[mask].total_bounds


def get_panel_center(gdf, hemi, view):
    """Get center point for a specific panel."""
    bounds = get_panel_bounds(gdf, hemi, view)
    return ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)


class TestPositionBrainParameters:
    def test_default_values(self):
        pos = position_brain()
        assert pos.rows == "hemi"
        assert pos.cols == "view"
        assert pos.nrow is None
        assert pos.ncol is None
        assert pos.views is None
        assert pos.spacing == 0.1


class TestVariableLayout:
    @pytest.fixture
    def atlas_gdf(self):
        return dk().data.ggseg.copy()

    def test_default_layout_separates_hemis_vertically(self, atlas_gdf):
        pos = position_brain()
        result = pos.apply(atlas_gdf)

        left_center = get_panel_center(result, "left", "lateral")
        right_center = get_panel_center(result, "right", "lateral")

        assert left_center[1] > right_center[1]

    def test_default_layout_separates_views_horizontally(self, atlas_gdf):
        pos = position_brain()
        result = pos.apply(atlas_gdf)

        lateral_center = get_panel_center(result, "left", "lateral")
        medial_center = get_panel_center(result, "left", "medial")

        assert lateral_center[0] < medial_center[0]

    def test_swapped_layout_views_as_rows(self, atlas_gdf):
        pos = position_brain(rows="view", cols="hemi")
        result = pos.apply(atlas_gdf)

        lateral_center = get_panel_center(result, "left", "lateral")
        medial_center = get_panel_center(result, "left", "medial")

        assert lateral_center[1] > medial_center[1]

    def test_single_column_stacks_all_panels_vertically(self, atlas_gdf):
        pos = position_brain(rows="hemi", cols=None)
        result = pos.apply(atlas_gdf)

        left_center = get_panel_center(result, "left", None)
        right_center = get_panel_center(result, "right", None)

        assert left_center[1] > right_center[1]


class TestGridLayout:
    @pytest.fixture
    def atlas_gdf(self):
        return dk().data.ggseg.copy()

    def test_ncol_2_with_2_views_creates_two_columns(self, atlas_gdf):
        pos = position_brain(ncol=2, views=["lateral", "medial"])
        result = pos.apply(atlas_gdf)

        lateral_left = get_panel_bounds(result, "left", "lateral")[0]
        medial_left = get_panel_bounds(result, "left", "medial")[0]

        assert lateral_left == pytest.approx(0, abs=1)
        assert medial_left > lateral_left + 100

    def test_nrow_1_creates_single_row(self, atlas_gdf):
        pos = position_brain(nrow=1, views=["lateral", "medial"])
        result = pos.apply(atlas_gdf)

        panels = [("left", "lateral"), ("left", "medial"),
                  ("right", "lateral"), ("right", "medial")]
        bottom_edges = [get_panel_bounds(result, h, v)[1] for h, v in panels]

        assert all(y == pytest.approx(0, abs=1) for y in bottom_edges)

    def test_ncol_1_with_2_views_stacks_vertically(self, atlas_gdf):
        pos = position_brain(ncol=1, views=["lateral", "medial"])
        result = pos.apply(atlas_gdf)

        panels = [("left", "lateral"), ("left", "medial"),
                  ("right", "lateral"), ("right", "medial")]
        left_edges = [get_panel_bounds(result, h, v)[0] for h, v in panels]

        assert all(x == pytest.approx(0, abs=1) for x in left_edges)

    def test_grid_respects_panel_count(self, atlas_gdf):
        pos = position_brain(ncol=2, views=["lateral", "medial"])
        result = pos.apply(atlas_gdf)

        n_unique_views = len(result["view"].unique())
        n_unique_hemis = len(result["hemi"].unique())
        n_panels = n_unique_views * n_unique_hemis

        assert n_panels == 4


class TestFiltering:
    @pytest.fixture
    def atlas_gdf(self):
        return dk().data.ggseg.copy()

    def test_views_filters_to_specified_views(self, atlas_gdf):
        pos = position_brain(views=["lateral"])
        result = pos.apply(atlas_gdf)

        assert "medial" not in result["view"].values
        assert "lateral" in result["view"].values

    def test_views_ordering_affects_grid_layout(self, atlas_gdf):
        pos_lateral_first = position_brain(nrow=1, views=["lateral", "medial"])
        pos_medial_first = position_brain(nrow=1, views=["medial", "lateral"])

        result_lf = pos_lateral_first.apply(atlas_gdf.copy())
        result_mf = pos_medial_first.apply(atlas_gdf.copy())

        lat_x_lf = get_panel_center(result_lf, "left", "lateral")[0]
        med_x_lf = get_panel_center(result_lf, "left", "medial")[0]

        lat_x_mf = get_panel_center(result_mf, "left", "lateral")[0]
        med_x_mf = get_panel_center(result_mf, "left", "medial")[0]

        assert lat_x_lf < med_x_lf
        assert lat_x_mf > med_x_mf

    def test_views_filters_multiple_views(self, atlas_gdf):
        pos = position_brain(views=["lateral", "medial"])
        result = pos.apply(atlas_gdf)

        assert set(result["view"].unique()) == {"lateral", "medial"}
        assert "superior" not in result["view"].values


class TestSpacing:
    @pytest.fixture
    def atlas_gdf(self):
        return dk().data.ggseg.copy()

    def test_larger_spacing_increases_panel_distance(self, atlas_gdf):
        pos_small = position_brain(spacing=0.1)
        pos_large = position_brain(spacing=0.5)

        result_small = pos_small.apply(atlas_gdf.copy())
        result_large = pos_large.apply(atlas_gdf.copy())

        def get_view_distance(gdf):
            lat = get_panel_center(gdf, "left", "lateral")[0]
            med = get_panel_center(gdf, "left", "medial")[0]
            return abs(med - lat)

        dist_small = get_view_distance(result_small)
        dist_large = get_view_distance(result_large)

        assert dist_large > dist_small

    def test_zero_spacing_panels_adjacent(self, atlas_gdf):
        pos = position_brain(spacing=0.0)
        result = pos.apply(atlas_gdf)

        lateral_bounds = get_panel_bounds(result, "left", "lateral")
        medial_bounds = get_panel_bounds(result, "left", "medial")

        lateral_right_edge = lateral_bounds[2]
        medial_left_edge = medial_bounds[0]

        gap = medial_left_edge - lateral_right_edge
        assert gap < 1


class TestEdgeCases:
    def test_empty_geodataframe_returns_empty(self):
        atlas_gdf = dk().data.ggseg.copy()
        empty_gdf = atlas_gdf[atlas_gdf["hemi"] == "nonexistent"]

        pos = position_brain()
        result = pos.apply(empty_gdf)

        assert len(result) == 0

    def test_single_panel_normalized_to_origin(self):
        atlas_gdf = dk().data.ggseg.copy()
        single_panel = atlas_gdf[
            (atlas_gdf["hemi"] == "left") & (atlas_gdf["view"] == "lateral")
        ].copy()

        original_bounds = single_panel.total_bounds.copy()
        original_width = original_bounds[2] - original_bounds[0]
        original_height = original_bounds[3] - original_bounds[1]

        pos = position_brain()
        result = pos.apply(single_panel)

        result_bounds = result.total_bounds
        assert result_bounds[0] == pytest.approx(0, abs=0.01)
        assert result_bounds[1] == pytest.approx(0, abs=0.01)
        result_width = result_bounds[2] - result_bounds[0]
        result_height = result_bounds[3] - result_bounds[1]
        assert result_width == pytest.approx(original_width, rel=0.001)
        assert result_height == pytest.approx(original_height, rel=0.001)

    def test_preserves_all_regions(self):
        atlas_gdf = dk().data.ggseg.copy()
        original_count = len(atlas_gdf)

        pos = position_brain(ncol=2)
        result = pos.apply(atlas_gdf)

        assert len(result) == original_count

    def test_preserves_region_shapes(self):
        atlas_gdf = dk().data.ggseg.copy()
        original_areas = atlas_gdf.geometry.area.values

        pos = position_brain()
        result = pos.apply(atlas_gdf)

        result_areas = result.geometry.area.values
        np.testing.assert_array_almost_equal(original_areas, result_areas)
