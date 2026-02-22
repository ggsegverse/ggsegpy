from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

from ggsegpy.data import AtlasData, CorticalData, SubcorticalData, TractData
from ggsegpy.validation import validate_atlas, validate_atlas_type

if TYPE_CHECKING:
    import geopandas as gpd


@dataclass
class BrainAtlas:
    """Brain atlas container for 2D and 3D visualization.

    This is the core class in ggsegpy, equivalent to R's ggseg_atlas.
    It holds atlas metadata, region definitions, geometry data, and
    color palettes.

    Parameters
    ----------
    atlas
        Short name for the atlas (e.g., 'dk', 'aseg', 'tracula').
    type
        Atlas type: 'cortical', 'subcortical', or 'tract'.
    core
        DataFrame with required columns: hemi, region, label.
        One row per unique region.
    data
        Type-specific data container (CorticalData, SubcorticalData,
        or TractData).
    palette
        Named dict mapping labels to hex colors.

    Examples
    --------
    >>> from ggsegpy import dk
    >>> atlas = dk()
    >>> print(atlas)
    >>> atlas.labels[:5]
    >>> atlas.to_dataframe()
    """

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

    def __repr__(self) -> str:
        return self._format_repr()

    def __str__(self) -> str:
        return self._format_repr()

    def _format_repr(self) -> str:
        """Format atlas for display (like R's print.ggseg_atlas)."""
        lines = []
        lines.append(f"─── {self.atlas} ggseg atlas ───")
        lines.append(f"Type: {self.type}")
        lines.append(f"Regions: {len(self.regions)}")
        lines.append(f"Hemispheres: {', '.join(self.hemispheres)}")

        if hasattr(self.data, "ggseg") and self.data.ggseg is not None:
            views = self.data.ggseg["view"].unique().tolist()
            lines.append(f"Views: {', '.join(views)}")

        has_palette = bool(self.palette)
        has_2d = hasattr(self.data, "ggseg") and self.data.ggseg is not None
        has_3d = self._has_3d_data()

        lines.append(f"Palette: {'✓' if has_palette else '✗'}")
        lines.append(f"Rendering: {'✓' if has_2d else '✗'} ggseg (2D)")
        lines.append(f"           {'✓' if has_3d else '✗'} ggseg3d (3D)")

        return "\n".join(lines)

    def _has_3d_data(self) -> bool:
        """Check if atlas has 3D rendering data."""
        if hasattr(self.data, "ggseg3d") and self.data.ggseg3d is not None:
            return len(self.data.ggseg3d) > 0
        return False

    @property
    def labels(self) -> list[str]:
        """List of all region labels."""
        return self.core["label"].tolist()

    @property
    def regions(self) -> list[str]:
        """List of unique region names."""
        return self.core["region"].unique().tolist()

    @property
    def hemispheres(self) -> list[str]:
        """List of hemispheres in the atlas."""
        return self.core["hemi"].unique().tolist()

    def filter(
        self,
        hemi: str | list[str] | None = None,
        region: str | list[str] | None = None,
    ) -> BrainAtlas:
        """Filter atlas by hemisphere or region.

        Parameters
        ----------
        hemi
            Hemisphere(s) to keep: 'left', 'right', or list.
        region
            Region(s) to keep.

        Returns
        -------
        BrainAtlas
            Filtered atlas copy (preserves subclass type).
        """
        core = self.core.copy()

        if hemi is not None:
            hemis = [hemi] if isinstance(hemi, str) else hemi
            core = core[core["hemi"].isin(hemis)]

        if region is not None:
            regions = [region] if isinstance(region, str) else region
            core = core[core["region"].isin(regions)]

        # Preserve the subclass type (CorticalAtlas, SubcorticalAtlas, TractAtlas)
        return self.__class__(
            atlas=self.atlas,
            type=self.type,
            core=core,
            data=self.data,
            palette=self.palette,
        )

    def to_dataframe(self) -> gpd.GeoDataFrame:
        """Convert atlas to GeoDataFrame for plotting.

        Merges core metadata with geometry data, similar to R's
        as.data.frame.ggseg_atlas.

        Returns
        -------
        gpd.GeoDataFrame
            Combined data with geometry, hemi, region, label, view,
            and atlas metadata columns.
        """
        import geopandas as gpd

        if not hasattr(self.data, "ggseg") or self.data.ggseg is None:
            raise ValueError("Atlas has no 2D geometry data")

        sf = self.data.ggseg.copy()

        core_cols = ["hemi", "region"]
        for col in core_cols:
            if col in sf.columns:
                del sf[col]

        result = sf.merge(self.core, on="label", how="left")

        result["atlas"] = self.atlas
        result["type"] = self.type

        if self.palette:
            result["colour"] = result["label"].map(self.palette)

        return gpd.GeoDataFrame(result, geometry="geometry")

    def plot(self, **kwargs: Any):
        """Quick plot of the atlas.

        Parameters
        ----------
        **kwargs
            Passed to geom_brain().

        Returns
        -------
        ggplot
            A plotnine ggplot object.
        """
        from plotnine import ggplot

        from ggsegpy.geom_brain import geom_brain

        return ggplot() + geom_brain(atlas=self, **kwargs)


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
