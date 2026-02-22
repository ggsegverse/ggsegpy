from __future__ import annotations

from plotnine import (
    element_blank,
    element_rect,
    element_text,
    theme,
)


def theme_brain(
    text_size: int = 12,
    text_family: str = "monospace",
) -> theme:
    """Clean theme for brain plots with minimal axes.

    Parameters
    ----------
    text_size
        Base text size. Default is 12.
    text_family
        Font family. Default is 'monospace'.

    Returns
    -------
    theme
        A plotnine theme object.
    """
    return theme(
        text=element_text(
            family=text_family,
            size=text_size,
            color="darkgrey",
        ),
        plot_background=element_rect(fill="white", color=None),
        panel_background=element_rect(fill="white", color=None),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        axis_text=element_blank(),
        axis_title=element_blank(),
        axis_ticks=element_blank(),
        axis_line=element_blank(),
        legend_background=element_rect(fill="white", color=None),
        legend_key=element_rect(fill="white", color=None),
        legend_position="right",
        legend_title=element_text(size=text_size),
        legend_text=element_text(size=text_size - 2),
        strip_background=element_rect(fill="white", color=None),
        strip_text=element_text(size=text_size),
    )


def theme_darkbrain(
    text_size: int = 12,
    text_family: str = "monospace",
) -> theme:
    """Dark theme for brain plots.

    Parameters
    ----------
    text_size
        Base text size. Default is 12.
    text_family
        Font family. Default is 'monospace'.

    Returns
    -------
    theme
        A plotnine theme object.
    """
    return theme_custombrain(
        plot_background="#222222",
        text_colour="white",
        text_size=text_size,
        text_family=text_family,
    )


def theme_custombrain(
    plot_background: str = "white",
    text_colour: str = "darkgrey",
    text_size: int = 12,
    text_family: str = "monospace",
) -> theme:
    """Customizable theme for brain plots.

    Parameters
    ----------
    plot_background
        Background color. Default is 'white'.
    text_colour
        Text color. Default is 'darkgrey'.
    text_size
        Base text size. Default is 12.
    text_family
        Font family. Default is 'monospace'.

    Returns
    -------
    theme
        A plotnine theme object.
    """
    return theme(
        text=element_text(
            family=text_family,
            size=text_size,
            color=text_colour,
        ),
        plot_background=element_rect(fill=plot_background, color=None),
        panel_background=element_rect(fill=plot_background, color=None),
        panel_grid_major=element_blank(),
        panel_grid_minor=element_blank(),
        panel_border=element_blank(),
        axis_text=element_blank(),
        axis_title=element_blank(),
        axis_ticks=element_blank(),
        axis_line=element_blank(),
        legend_background=element_rect(fill=plot_background, color=None),
        legend_key=element_rect(fill=plot_background, color=None),
        legend_position="right",
        legend_title=element_text(size=text_size, color=text_colour),
        legend_text=element_text(size=text_size - 2, color=text_colour),
        strip_background=element_rect(fill=plot_background, color=None),
        strip_text=element_text(size=text_size, color=text_colour),
    )


def theme_brain_void(
    text_size: int = 12,
    text_family: str = "monospace",
) -> theme:
    """Void theme with no axes, background, or grid.

    Parameters
    ----------
    text_size
        Base text size. Default is 12.
    text_family
        Font family. Default is 'monospace'.

    Returns
    -------
    theme
        A plotnine theme object.
    """
    return theme(
        text=element_text(family=text_family, size=text_size),
        line=element_blank(),
        rect=element_blank(),
        axis_text=element_blank(),
        axis_title=element_blank(),
        axis_ticks=element_blank(),
        panel_grid=element_blank(),
        panel_border=element_blank(),
        plot_background=element_blank(),
        panel_background=element_blank(),
        legend_background=element_blank(),
        legend_key=element_blank(),
        strip_background=element_blank(),
    )
