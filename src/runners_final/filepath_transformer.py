from pathlib import Path
from typing import Optional
from matplotlib.ticker import MaxNLocator

from src.pipeline.mask_processing import *
from src.pipeline.properties_computation.angles_at_intersections import *


SRC_DIR = next((p for p in Path(__file__).resolve().parents if p.name == "src"), None, )
if SRC_DIR is None:
    raise FileNotFoundError("Could not locate the 'src' directory from this script.")

DEFAULT_INPUT_PATH = SRC_DIR / "data/input"
DEFAULT_OUTPUT_PATH = SRC_DIR / "data/graphs"
DEFAULT_HTML_PATH = SRC_DIR / "data/html"
DEFAULT_PLOT_PATH = SRC_DIR / "data/plots"
DEFAULT_MASK_PATH = SRC_DIR / "data/mask"


def transform_filepath(filepath: Optional[str], default_dir: Path, must_exist: bool = False) -> Optional[str]:
    """
    Convert a path argument into an absolute path string:
    - Absolute path (or '~' home path): use as-is.
    - Only a filename (no slashes/subdirectories): prepend `default_dir`.
    - Relative paths with directories (e.g., './file.nii', 'sub/file.nii'): raise ValueError.
    """
    if filepath is None:
        return None

    has_separator = "/" in filepath or "\\" in filepath
    path = Path(filepath).expanduser()

    if path.is_absolute():
        final_path = path.resolve()
    elif not has_separator and path.name == filepath and filepath not in (".", ".."):
        final_path = (default_dir / path).resolve()
    else:
        raise ValueError(
            f"Unsupported relative path '{filepath}'. "
            "Provide either an absolute path or only a filename."
        )

    if must_exist:
        if not final_path.is_file():
            raise FileNotFoundError(f"Input file not found: {final_path}")
    else:
        final_path.parent.mkdir(parents=True, exist_ok=True)

    return str(final_path)


def plot_len_degree(list_distances, title: str, output_path, x_axis_label: Optional[str] = None,):
    if not len(list_distances):
        print(f"No distances to plot for {title}")
        return

    min_dist = int(np.floor(min(list_distances)))
    max_edge = int(np.floor(max(list_distances))) + 1

    # --- 1. Define custom bins ---
    if max_edge > 91:
        fine_bins = np.arange(min_dist, 91, 1) if min_dist < 90 else np.array([], dtype=int)
        coarse_start = max(90, min_dist)
        coarse_bins = np.unique(np.ceil(np.linspace(coarse_start, max_edge, 8)).astype(int))
        custom_bins = np.unique(np.concatenate((fine_bins, coarse_bins)))
    else:
        custom_bins = np.arange(min_dist, max_edge + 1, 1)

    # --- 2. Calculate histogram ---
    counts, bin_edges = np.histogram(list_distances, bins=custom_bins)
    n_bars = len(counts)
    x_positions = np.arange(n_bars)

    # --- 3. Adapt layout & bar sizing to the number of bars ---
    fig_width = float(np.clip(4.5 + 0.25 * n_bars, 6.0, 16.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    if n_bars == 1:
        bar_width = 0.45
    elif n_bars <= 5:
        bar_width = 0.65
    elif n_bars <= 20:
        bar_width = 0.80
    else:
        bar_width = 0.90

    ax.set_axisbelow(True)  # Keep grid behind bars
    bars = ax.bar(
        x_positions,
        counts,
        width=bar_width,
        color="royalblue",
        edgecolor="black",
        alpha=0.85,
    )

    # --- 4. Adapt x-axis limits so few bars stay centered and proportional ---
    if n_bars == 1:
        ax.set_xlim(-1.2, 1.2)
    elif n_bars <= 5:
        ax.set_xlim(-0.8, n_bars - 0.2)
    else:
        ax.set_xlim(-0.6, n_bars - 0.4)

    # --- 5. Format X-Axis Labels (show every tick) ---
    ticks_to_show = np.arange(n_bars)
    labels_to_show = []

    for i, edge in enumerate(bin_edges[:-1]):
        next_edge = bin_edges[i + 1]
        is_range = edge >= 90 and (next_edge - edge > 1)
        labels_to_show.append(f"{edge}-{next_edge}" if is_range else str(edge))

    has_long_labels = any(len(lbl) > 3 for lbl in labels_to_show)
    if n_bars > 40:
        rotation, ha, tick_fontsize = 45, "center", 7
    elif n_bars > 12 or has_long_labels:
        rotation, ha, tick_fontsize = 45, "right", 8.5
    else:
        rotation, ha, tick_fontsize = 0, "center", 9.5

    ax.set_xticks(ticks_to_show)
    ax.set_xticklabels(labels_to_show, rotation=rotation, ha=ha, fontsize=tick_fontsize)

    # --- 6. Y-axis headroom, integer ticks & value labels ---
    max_count = int(counts.max()) if counts.size else 1
    ax.set_ylim(0, max_count * 1.15)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    text_fontsize = 9.5 if n_bars <= 25 else 7.5
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(
                str(int(height)),
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=text_fontsize,
                color="black",
            )

    # --- 7. Clean styling ---
    ax.set_title(title, fontsize=14, pad=12)
    if x_axis_label:
        ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    # plt.show()
    plt.close(fig)
