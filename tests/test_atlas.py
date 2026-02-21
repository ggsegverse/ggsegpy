from __future__ import annotations

import pandas as pd
import pytest

from ggsegpy import BrainAtlas, CorticalAtlas, dk
from ggsegpy.data import CorticalData
from ggsegpy.validation import validate_core


class TestValidateCore:
    def test_valid_core(self):
        core = pd.DataFrame({
            "hemi": ["left", "right"],
            "region": ["bankssts", "bankssts"],
            "label": ["lh_bankssts", "rh_bankssts"],
        })
        errors = validate_core(core)
        assert errors == []

    def test_missing_columns(self):
        core = pd.DataFrame({"hemi": ["left"], "region": ["bankssts"]})
        errors = validate_core(core)
        assert len(errors) == 1
        assert "label" in errors[0]

    def test_invalid_hemi(self):
        core = pd.DataFrame({
            "hemi": ["invalid"],
            "region": ["bankssts"],
            "label": ["invalid_bankssts"],
        })
        errors = validate_core(core)
        assert len(errors) == 1
        assert "invalid" in errors[0]

    def test_duplicate_labels(self):
        core = pd.DataFrame({
            "hemi": ["left", "left"],
            "region": ["bankssts", "cuneus"],
            "label": ["duplicate", "duplicate"],
        })
        errors = validate_core(core)
        assert len(errors) == 1
        assert "duplicate" in errors[0]


class TestBrainAtlas:
    def test_labels_property(self):
        atlas = dk()
        labels = atlas.labels
        assert isinstance(labels, list)
        assert "lh_bankssts" in labels
        assert "rh_bankssts" in labels

    def test_regions_property(self):
        atlas = dk()
        regions = atlas.regions
        assert isinstance(regions, list)
        assert len(regions) > 0

    def test_hemispheres_property(self):
        atlas = dk()
        hemis = atlas.hemispheres
        assert set(hemis) == {"left", "right"}

    def test_filter_by_hemi(self):
        atlas = dk()
        filtered = atlas.filter(hemi="left")
        assert all(h == "left" for h in filtered.core["hemi"])

    def test_filter_by_region(self):
        atlas = dk()
        filtered = atlas.filter(region="bankssts")
        assert all(r == "bankssts" for r in filtered.core["region"])


class TestCorticalAtlas:
    def test_dk_atlas_creation(self):
        atlas = dk()
        assert isinstance(atlas, CorticalAtlas)
        assert atlas.atlas == "dk"
        assert atlas.type == "cortical"

    def test_dk_has_core_data(self):
        atlas = dk()
        assert len(atlas.core) > 0
        assert "hemi" in atlas.core.columns
        assert "region" in atlas.core.columns
        assert "label" in atlas.core.columns

    def test_dk_has_geometry(self):
        atlas = dk()
        assert atlas.data.ggseg is not None
        assert "geometry" in atlas.data.ggseg.columns

    def test_dk_has_palette(self):
        atlas = dk()
        assert len(atlas.palette) > 0
        assert "lh_bankssts" in atlas.palette
