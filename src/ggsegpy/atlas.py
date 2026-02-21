from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from ggsegpy.data import AtlasData, CorticalData, SubcorticalData, TractData
from ggsegpy.validation import validate_atlas, validate_atlas_type


@dataclass
class BrainAtlas:
    atlas: str
    type: Literal["cortical", "subcortical", "tract"]
    core: pd.DataFrame
    data: AtlasData
    palette: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.type = validate_atlas_type(self.type)
        errors, warnings = validate_atlas(self)
        if errors:
            raise ValueError(f"Invalid atlas: {errors}")
        for warning in warnings:
            import warnings as w

            w.warn(warning, UserWarning, stacklevel=2)

    @property
    def labels(self) -> list[str]:
        return self.core["label"].tolist()

    @property
    def regions(self) -> list[str]:
        return self.core["region"].unique().tolist()

    @property
    def hemispheres(self) -> list[str]:
        return self.core["hemi"].unique().tolist()

    def filter(
        self,
        hemi: str | list[str] | None = None,
        region: str | list[str] | None = None,
    ) -> BrainAtlas:
        core = self.core.copy()

        if hemi is not None:
            hemis = [hemi] if isinstance(hemi, str) else hemi
            core = core[core["hemi"].isin(hemis)]

        if region is not None:
            regions = [region] if isinstance(region, str) else region
            core = core[core["region"].isin(regions)]

        return BrainAtlas(
            atlas=self.atlas,
            type=self.type,
            core=core,
            data=self.data,
            palette=self.palette,
        )


@dataclass
class CorticalAtlas(BrainAtlas):
    data: CorticalData = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.data, CorticalData):
            raise TypeError("CorticalAtlas requires CorticalData")
        super().__post_init__()


@dataclass
class SubcorticalAtlas(BrainAtlas):
    data: SubcorticalData = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.data, SubcorticalData):
            raise TypeError("SubcorticalAtlas requires SubcorticalData")
        super().__post_init__()


@dataclass
class TractAtlas(BrainAtlas):
    data: TractData = field(default=None)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if not isinstance(self.data, TractData):
            raise TypeError("TractAtlas requires TractData")
        super().__post_init__()
