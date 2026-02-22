from __future__ import annotations

import pandas as pd
import pytest

from ggsegpy import (
    atlas_core_add,
    atlas_labels,
    atlas_palette,
    atlas_region_contextual,
    atlas_region_keep,
    atlas_region_remove,
    atlas_region_rename,
    atlas_regions,
    atlas_sf,
    atlas_type,
    atlas_view_gather,
    atlas_view_keep,
    atlas_view_remove,
    atlas_view_remove_region,
    atlas_view_remove_small,
    atlas_view_reorder,
    atlas_views,
    dk,
)


class TestAtlasAccessors:
    def test_atlas_regions_returns_sorted_unique_regions(self):
        atlas = dk()
        regions = atlas_regions(atlas)

        assert isinstance(regions, list)
        assert len(regions) > 0
        assert regions == sorted(regions)
        assert len(regions) == len(set(regions))

    def test_atlas_labels_returns_sorted_unique_labels(self):
        atlas = dk()
        labels = atlas_labels(atlas)

        assert isinstance(labels, list)
        assert len(labels) > 0
        assert labels == sorted(labels)
        assert len(labels) == len(set(labels))

    def test_atlas_type_returns_correct_type(self):
        atlas = dk()
        result = atlas_type(atlas)

        assert result == "cortical"

    def test_atlas_views_returns_view_list(self):
        atlas = dk()
        views = atlas_views(atlas)

        assert isinstance(views, list)
        assert "lateral" in views
        assert "medial" in views

    def test_atlas_palette_returns_dict_copy(self):
        atlas = dk()
        palette = atlas_palette(atlas)

        assert isinstance(palette, dict)
        assert len(palette) > 0
        palette["test"] = "#000000"
        assert "test" not in atlas.palette

    def test_atlas_sf_returns_geodataframe(self):
        atlas = dk()
        sf = atlas_sf(atlas)

        assert "geometry" in sf.columns
        assert "label" in sf.columns
        assert "colour" in sf.columns


class TestAtlasRegionRemove:
    def test_removes_matching_regions(self):
        atlas = dk()
        original_count = len(atlas_regions(atlas))

        filtered = atlas_region_remove(atlas, "banks")

        new_regions = atlas_regions(filtered)
        assert len(new_regions) < original_count
        assert not any("banks" in r.lower() for r in new_regions)

    def test_removes_from_palette(self):
        atlas = dk()
        filtered = atlas_region_remove(atlas, "banks")

        assert not any("banks" in k.lower() for k in filtered.palette.keys())

    def test_removes_from_sf_data(self):
        atlas = dk()
        filtered = atlas_region_remove(atlas, "banks")

        sf = atlas_sf(filtered)
        labels = sf["label"].dropna().tolist()
        assert not any("banks" in lbl.lower() for lbl in labels)

    def test_match_on_label(self):
        atlas = dk()
        filtered = atlas_region_remove(atlas, "lh_", match_on="label")

        labels = atlas_labels(filtered)
        assert not any(lbl.startswith("lh_") for lbl in labels)

    def test_invalid_match_on_raises(self):
        atlas = dk()
        with pytest.raises(ValueError, match="match_on must be"):
            atlas_region_remove(atlas, "banks", match_on="invalid")


class TestAtlasRegionContextual:
    def test_removes_from_core_but_keeps_sf(self):
        atlas = dk()
        original_sf_count = len(atlas.data.ggseg)

        filtered = atlas_region_contextual(atlas, "banks")

        new_regions = atlas_regions(filtered)
        assert not any("banks" in r.lower() for r in new_regions)

        new_sf_count = len(filtered.data.ggseg)
        assert new_sf_count == original_sf_count

    def test_removes_from_palette(self):
        atlas = dk()
        filtered = atlas_region_contextual(atlas, "banks")

        assert not any("banks" in k.lower() for k in filtered.palette.keys())


class TestAtlasRegionRename:
    def test_renames_matching_regions(self):
        atlas = dk()
        renamed = atlas_region_rename(atlas, "superior", "sup.")

        regions = atlas_regions(renamed)
        assert any("sup." in r for r in regions)
        assert not any("superior" in r.lower() for r in regions)

    def test_callable_replacement(self):
        atlas = dk()
        renamed = atlas_region_rename(atlas, ".*", lambda x: x.upper())

        regions = atlas_regions(renamed)
        assert all(r.isupper() for r in regions)

    def test_preserves_labels(self):
        atlas = dk()
        original_labels = atlas_labels(atlas)

        renamed = atlas_region_rename(atlas, "superior", "sup.")

        assert atlas_labels(renamed) == original_labels


class TestAtlasRegionKeep:
    def test_keeps_only_matching_regions(self):
        atlas = dk()
        filtered = atlas_region_keep(atlas, "frontal")

        regions = atlas_regions(filtered)
        assert len(regions) > 0
        assert all("frontal" in r.lower() for r in regions)

    def test_keeps_sf_for_continuity(self):
        atlas = dk()
        original_sf_count = len(atlas.data.ggseg)

        filtered = atlas_region_keep(atlas, "frontal")

        new_sf_count = len(filtered.data.ggseg)
        assert new_sf_count == original_sf_count


class TestAtlasCoreAdd:
    def test_adds_metadata_columns(self):
        atlas = dk()
        meta = pd.DataFrame({
            "region": ["precentral", "postcentral"],
            "lobe": ["frontal", "parietal"],
        })

        enriched = atlas_core_add(atlas, meta, by="region")

        assert "lobe" in enriched.core.columns
        precentral_row = enriched.core[enriched.core["region"] == "precentral"]
        assert precentral_row["lobe"].iloc[0] == "frontal"


class TestAtlasViewRemove:
    def test_removes_matching_views(self):
        atlas = dk()
        filtered = atlas_view_remove(atlas, "lateral")

        views = atlas_views(filtered)
        assert "lateral" not in views

    def test_removes_multiple_views(self):
        atlas = dk()
        filtered = atlas_view_remove(atlas, ["lateral", "medial"])

        views = atlas_views(filtered)
        assert "lateral" not in views
        assert "medial" not in views


class TestAtlasViewKeep:
    def test_keeps_only_matching_views(self):
        atlas = dk()
        filtered = atlas_view_keep(atlas, "lateral")

        views = atlas_views(filtered)
        assert views == ["lateral"]

    def test_keeps_multiple_views(self):
        atlas = dk()
        filtered = atlas_view_keep(atlas, ["lateral", "medial"])

        views = atlas_views(filtered)
        assert set(views) == {"lateral", "medial"}


class TestAtlasSubclassPreservation:
    def test_region_remove_preserves_subclass(self):
        atlas = dk()
        filtered = atlas_region_remove(atlas, "banks")

        assert type(filtered).__name__ == "CorticalAtlas"

    def test_region_keep_preserves_subclass(self):
        atlas = dk()
        filtered = atlas_region_keep(atlas, "frontal")

        assert type(filtered).__name__ == "CorticalAtlas"

    def test_view_remove_preserves_subclass(self):
        atlas = dk()
        filtered = atlas_view_remove(atlas, "lateral")

        assert type(filtered).__name__ == "CorticalAtlas"


class TestAtlasViewRemoveRegion:
    def test_removes_region_geometry_from_sf(self):
        atlas = dk()
        original_count = len(atlas.data.ggseg)

        filtered = atlas_view_remove_region(atlas, "banks")

        new_count = len(filtered.data.ggseg)
        assert new_count < original_count

        labels_in_sf = filtered.data.ggseg["label"].dropna().tolist()
        assert not any("banks" in lbl.lower() for lbl in labels_in_sf)

    def test_preserves_core_and_palette(self):
        atlas = dk()
        original_regions = atlas_regions(atlas)
        original_palette_size = len(atlas.palette)

        filtered = atlas_view_remove_region(atlas, "banks")

        assert atlas_regions(filtered) == original_regions
        assert len(filtered.palette) == original_palette_size

    def test_scoped_to_specific_view(self):
        atlas = dk()
        lateral_banks = atlas.data.ggseg[
            (atlas.data.ggseg["view"] == "lateral") &
            (atlas.data.ggseg["label"].str.contains("banks", case=False, na=False))
        ]
        original_lateral_banks_count = len(lateral_banks)

        filtered = atlas_view_remove_region(atlas, "banks", views="lateral")

        lateral_sf = filtered.data.ggseg[filtered.data.ggseg["view"] == "lateral"]
        lateral_banks_after = lateral_sf[
            lateral_sf["label"].str.contains("banks", case=False, na=False)
        ]
        assert len(lateral_banks_after) == 0

        medial_sf = filtered.data.ggseg[filtered.data.ggseg["view"] == "medial"]
        medial_banks = medial_sf[
            medial_sf["label"].str.contains("banks", case=False, na=False)
        ]
        assert len(medial_banks) > 0 or original_lateral_banks_count > 0


class TestAtlasViewRemoveSmall:
    def test_removes_small_geometries(self):
        atlas = dk()
        areas = atlas.data.ggseg.geometry.area
        median_area = areas.median()

        filtered = atlas_view_remove_small(atlas, min_area=median_area)

        remaining_areas = filtered.data.ggseg.geometry.area
        core_labels = set(atlas.core["label"])
        is_core = filtered.data.ggseg["label"].isin(core_labels)
        core_areas = remaining_areas[is_core]

        assert (core_areas >= median_area).all() or len(core_areas) == 0

    def test_preserves_context_geometry(self):
        atlas = dk()
        huge_area = atlas.data.ggseg.geometry.area.max() * 10

        filtered = atlas_view_remove_small(atlas, min_area=huge_area)

        context_labels = filtered.data.ggseg[
            ~filtered.data.ggseg["label"].isin(atlas.core["label"])
        ]
        assert len(context_labels) >= 0


class TestAtlasViewGather:
    def test_repositions_views(self):
        atlas = dk()
        original_bounds = atlas.data.ggseg.total_bounds

        gathered = atlas_view_gather(atlas)

        new_bounds = gathered.data.ggseg.total_bounds
        assert new_bounds is not None
        assert len(gathered.data.ggseg) == len(atlas.data.ggseg)

    def test_preserves_subclass(self):
        atlas = dk()
        gathered = atlas_view_gather(atlas)

        assert type(gathered).__name__ == "CorticalAtlas"


class TestAtlasViewReorder:
    def test_reorders_views(self):
        atlas = dk()
        views = atlas_views(atlas)

        reversed_order = list(reversed(views))
        reordered = atlas_view_reorder(atlas, reversed_order)

        assert len(reordered.data.ggseg) == len(atlas.data.ggseg)

    def test_appends_unspecified_views(self):
        atlas = dk()
        views = atlas_views(atlas)

        partial_order = views[:1]
        reordered = atlas_view_reorder(atlas, partial_order)

        new_views = atlas_views(reordered)
        assert set(new_views) == set(views)

    def test_preserves_subclass(self):
        atlas = dk()
        views = atlas_views(atlas)

        reordered = atlas_view_reorder(atlas, views)

        assert type(reordered).__name__ == "CorticalAtlas"


class TestRegexEscaping:
    def test_view_keep_with_special_characters(self):
        atlas = dk()
        result = atlas_view_keep(atlas, "lateral")
        assert len(result.data.ggseg) > 0

        result_dot = atlas_view_keep(atlas, "lat.ral")
        assert len(result_dot.data.ggseg) == 0

    def test_view_remove_with_parentheses(self):
        atlas = dk()
        original_count = len(atlas.data.ggseg)

        result = atlas_view_remove(atlas, "lateral(fake)")
        assert len(result.data.ggseg) == original_count

    def test_view_remove_region_escapes_pattern(self):
        atlas = dk()
        original_count = len(atlas.data.ggseg)

        result = atlas_view_remove_region(atlas, "banks", views="lateral(fake)")
        assert len(result.data.ggseg) == original_count


class TestGapValidation:
    def test_negative_gap_raises(self):
        atlas = dk()
        with pytest.raises(ValueError, match="gap must be between"):
            atlas_view_gather(atlas, gap=-0.5)

    def test_large_gap_raises(self):
        atlas = dk()
        with pytest.raises(ValueError, match="gap must be between"):
            atlas_view_gather(atlas, gap=3.0)

    def test_valid_gap_succeeds(self):
        atlas = dk()
        result = atlas_view_gather(atlas, gap=0.5)
        assert len(result.data.ggseg) == len(atlas.data.ggseg)

    def test_zero_gap_succeeds(self):
        atlas = dk()
        result = atlas_view_gather(atlas, gap=0.0)
        assert len(result.data.ggseg) == len(atlas.data.ggseg)

    def test_reorder_validates_gap(self):
        atlas = dk()
        with pytest.raises(ValueError, match="gap must be between"):
            atlas_view_reorder(atlas, ["lateral"], gap=-1.0)


class TestMatchOnValidation:
    def test_invalid_match_on_in_region_remove(self):
        atlas = dk()
        with pytest.raises(ValueError, match="match_on must be"):
            atlas_region_remove(atlas, "banks", match_on="invalid")

    def test_invalid_match_on_in_region_keep(self):
        atlas = dk()
        with pytest.raises(ValueError, match="match_on must be"):
            atlas_region_keep(atlas, "banks", match_on="foo")

    def test_invalid_match_on_in_view_remove_region(self):
        atlas = dk()
        with pytest.raises(ValueError, match="match_on must be"):
            atlas_view_remove_region(atlas, "banks", match_on="bar")
