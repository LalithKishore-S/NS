"""Fixed family -> color mapping (same colors in every figure)."""
FAMILY_COLOR = {"ER": "#2a78d6", "BA": "#eb6834", "WS": "#1baf7a", "SBM": "#eda100"}
FAMILY_ORDER = ["ER", "BA", "WS", "SBM"]


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.tick_params(colors="#555")
