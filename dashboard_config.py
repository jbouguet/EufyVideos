# dashboard_config.py

"""
Centralized configuration for both static and interactive dashboards.
Ensures consistent styling and behavior across all dashboard implementations.
"""

from typing import Any, Dict

import plotly.express as px

from config import Config


class DashboardConfig:
    """Centralized configuration for dashboard styling and behavior"""

    # Color palettes for plotting
    COLOR_PALETTES = {
        "light24": px.colors.qualitative.Light24,
        "alphabet": px.colors.qualitative.Alphabet,
        "viridis": px.colors.sequential.Viridis,
        "custom": [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ],
        "colorblind": [
            "#0072B2",
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#D55E00",
            "#CC79A7",
            "#999999",
        ],
    }

    # Selected color palette for plotting
    SELECTED_PALETTE = "light24"

    # Figure height for plots
    FIGURE_HEIGHT = 800

    @classmethod
    def get_device_colors(cls) -> Dict[str, str]:
        """Get color mapping for devices based on their order"""
        devices = Config.get_all_devices()
        color_palette = cls.COLOR_PALETTES[cls.SELECTED_PALETTE]
        return {
            device: color_palette[i % len(color_palette)]
            for i, device in enumerate(devices)
        }

    @classmethod
    def get_figure_height(cls) -> int:
        """Get standard figure height"""
        return cls.FIGURE_HEIGHT

    @staticmethod
    def get_figure_config() -> Dict[str, Any]:
        """Default figure configuration"""
        return {
            "layout": {
                "margin": dict(l=30, r=30, t=10, b=30),
                "plot_bgcolor": "white",
                "paper_bgcolor": "white",
                "showlegend": True,
                "legend": dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                "barmode": "stack",
                "bargap": 0,
            },
            "axes": {
                "grid": dict(
                    showgrid=True, gridwidth=1, gridcolor="lightgrey", zeroline=False
                ),
                "yaxis": dict(rangemode="nonnegative", tickfont=dict(size=8)),
                "xaxis_date": dict(
                    dtick="D1",
                    tickformat="%a %Y/%m/%d",
                    tickangle=-90,
                    tickfont=dict(size=6),
                    rangeslider=dict(visible=False),
                    type="date",
                ),
            },
            "font": {"family": "var(--bs-font-sans-serif)", "size": 14},
        }

    @staticmethod
    def get_html_styles() -> Dict[str, Dict[str, str]]:
        """CSS styles for static HTML dashboard"""
        return {
            "container": {"padding": "0", "margin": "0", "width": "100%"},
            "title": {
                "font-family": "var(--bs-font-sans-serif)",
                "font-weight": "500",
                "font-size": "2rem",
                "line-height": "1.2",
                "margin": "1rem 0",
                "color": "var(--bs-body-color)",
                "text-align": "center",
            },
            "graph_container": {"margin": "0", "padding": "0"},
        }

    @staticmethod
    def get_dash_styles() -> Dict[str, Dict]:
        """Component styles for Dash dashboard"""
        return {
            "controls_card": {
                "style": {
                    "background-color": "rgb(245, 250, 255)",
                    "margin": "2px",
                    "border-radius": "5px",
                }
            },
            "controls_labels": {
                "style": {
                    "font-size": "12px",
                    "font-style": "italic",
                    "font-weight": "bold",
                    "color": "rgb(100, 100, 100)",
                }
            },
            "controls_text": {
                "style": {
                    "font-size": "12px",
                    "color": "rgb(100, 100, 100)",
                }
            },
            "controls_items": {
                "style": {
                    "font-size": "10px",
                }
            },
            "controls_spacing": {
                "style": {
                    "margin-right": "4px",
                    "margin-top": "4px",
                }
            },
            "container": {"style": {"padding": "0", "margin": "0"}},
            "title": {
                "className": "dashboard-title text-center",
                "style": {
                    "fontFamily": "var(--bs-font-sans-serif)",
                    "fontSize": "2rem",
                    "fontWeight": "500",
                    "lineHeight": "1.2",
                    "margin": "1rem 0",
                    "color": "var(--bs-body-color)",
                },
            },
            "graph": {"style": {"margin": "0", "padding": "0"}},
            "div_groups": {
                "style": {
                    # "bottom": "0",
                    "margin": "0px",
                    "padding": "0",
                    # "outline": "1px dashed green",
                }
            },
        }
