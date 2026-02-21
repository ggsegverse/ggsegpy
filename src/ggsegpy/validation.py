from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from ggsegpy.atlas import BrainAtlas


def validate_core(core: pd.DataFrame) -> list[str]:
    errors = []
    required = {"hemi", "region", "label"}
    missing = required - set(core.columns)
    if missing:
        errors.append(f"core missing required columns: {missing}")

    if "hemi" in core.columns:
        valid_hemis = {"left", "right", "midline", "subcort"}
        invalid = set(core["hemi"].unique()) - valid_hemis
        if invalid:
            errors.append(f"invalid hemi values: {invalid}")

    if "label" in core.columns and core["label"].duplicated().any():
        dups = core.loc[core["label"].duplicated(), "label"].tolist()
        errors.append(f"duplicate labels in core: {dups}")

    return errors


def validate_atlas_type(
    atlas_type: str,
) -> Literal["cortical", "subcortical", "tract"]:
    valid_types = {"cortical", "subcortical", "tract"}
    if atlas_type not in valid_types:
        raise ValueError(f"atlas type must be one of {valid_types}, got: {atlas_type}")
    return atlas_type  # type: ignore[return-value]


def validate_palette(palette: dict[str, str], labels: set[str]) -> list[str]:
    warnings = []
    palette_labels = set(palette.keys())

    missing_in_palette = labels - palette_labels
    if missing_in_palette:
        warnings.append(f"labels without palette colors: {missing_in_palette}")

    extra_in_palette = palette_labels - labels
    if extra_in_palette:
        warnings.append(f"palette colors for unknown labels: {extra_in_palette}")

    return warnings


def validate_atlas(atlas: BrainAtlas) -> tuple[list[str], list[str]]:
    errors = []
    warnings = []

    errors.extend(validate_core(atlas.core))

    if atlas.palette:
        labels = set(atlas.core["label"].unique())
        warnings.extend(validate_palette(atlas.palette, labels))

    return errors, warnings
