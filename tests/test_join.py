from __future__ import annotations

import pandas as pd
import pytest

from ggsegpy import brain_join, dk


class TestBrainJoin:
    def test_join_by_label(self):
        atlas = dk()
        data = pd.DataFrame({
            "label": ["lh_bankssts", "rh_bankssts"],
            "value": [0.5, 0.8],
        })
        result = brain_join(data, atlas)
        assert "value" in result.columns
        merged_values = result[result["label"].isin(data["label"])]["value"]
        assert not merged_values.isna().all()

    def test_join_by_region(self):
        atlas = dk()
        data = pd.DataFrame({
            "region": ["bankssts", "cuneus"],
            "value": [0.5, 0.8],
        })
        result = brain_join(data, atlas)
        assert "value" in result.columns

    def test_join_by_region_and_hemi(self):
        atlas = dk()
        data = pd.DataFrame({
            "region": ["bankssts", "bankssts"],
            "hemi": ["left", "right"],
            "value": [0.5, 0.8],
        })
        result = brain_join(data, atlas, by=["region", "hemi"])
        assert "value" in result.columns

    def test_unmatched_labels_warning(self):
        atlas = dk()
        data = pd.DataFrame({
            "label": ["nonexistent_region"],
            "value": [0.5],
        })
        with pytest.warns(UserWarning, match="not in atlas"):
            brain_join(data, atlas)

    def test_explicit_join_column(self):
        atlas = dk()
        data = pd.DataFrame({
            "my_label": ["lh_bankssts"],
            "value": [0.5],
        })
        result = brain_join(data, atlas, by="my_label")
        assert "value" in result.columns

    def test_missing_join_column_raises(self):
        atlas = dk()
        data = pd.DataFrame({
            "other_column": ["value"],
            "value": [0.5],
        })
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            brain_join(data, atlas)

    def test_facet_expansion(self):
        atlas = dk()
        data = pd.DataFrame({
            "region": ["precentral", "precentral"],
            "group": ["A", "B"],
            "value": [0.9, 0.3],
        })
        result = brain_join(data, atlas)
        assert "group" in result.columns
        assert set(result["group"].dropna().unique()) == {"A", "B"}
        n_atlas_rows = len(atlas.data.ggseg)
        n_groups = 2
        assert len(result) == n_atlas_rows * n_groups
