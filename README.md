# Filament Segmentation-to-Graph & Graph Analysis Pipeline

A two-stage Python command-line pipeline for converting 3D binary segmentation masks (`.nii`, or `.nrrd`) into topologically identical graph representations and analyzing them to detect Arp2/3 branch intersections, export 3D pick masks, and summarize network geometry.

## Pipeline Overview

1. **Stage 1 — Segmentation to Graph (`graph_construct.py`):** Loads a 3D binary mask, applies morphological closing, skeletonizes and classifies the resulting skeleton voxels, builds the topologically equivalent graph and stores it (as a `.pkl`). Optionally, an interactive HTML visualization is also stored.
2. **Stage 2 — Graph Analysis & Arp2/3 Detection (`compute_properties.py`):** Loads the graph, filters Arp2/3 candidates matching the `(70°, 110°, 180°)` angle signature, writes a 3D picks mask, plots intersection degree and subfilament length distributions, and prints descriptive statistics.

## Requirements

* Python 3.8+
* Core packages: `numpy`, `pandas`, `nibabel`, `pynrrd`, `scikit-image`, `matplotlib`, `networkx`, `pyvis`

```bash
pip install numpy pandas nibabel pynrrd scikit-image matplotlib networkx
```

## Shared File Path Rules

Both scripts resolve CLI path arguments through `filepath_transformer.py` using the same strict rules:

1. **Bare filename (no slashes):** Automatically resolved inside the project's default directory under `src/data`, containing separate folders for `input/`, `graphs/`, `html/`, `masks/` and `plots/`.
2. **Absolute path** (including `~` home paths): Used directly as provided.

> **Note:** Relative paths containing directory separators (such as `./mask.nii`, `../mask.nii`, or `folder/mask.nii`) are intentionally rejected with a `ValueError` to prevent working-directory ambiguity. Missing output directories are created automatically.

---

## Stage 1: Segmentation-to-Graph Generator

Converts a 3D binary segmentation mask into a graph representation (`.pkl`) and an optional HTML visualization.

### Processing Steps

1. **Load Mask:** Reads the binary segmentation mask (`.nii` or `.nrrd`).
2. **Morphological Closing:** Applies morphological closing using a spherical (`ball`) structuring element with a configurable integer radius.
3. **Skeletonization & Classification:** Extracts the 3D skeleton and finds endpoints and intersection voxels.
4. **Graph Construction:** Builds a topologically identical graph representation, writes it to `.pkl`, and optionally exports an HTML visualization.

### Usage

```bash
python graph_construct.py -i <input_mask> -o <output_graph> [-oh <output_html>] [-cr <closing_radius>]
```

### Command-Line Arguments

| Flag  | Long Flag | Required | Type | Default | Description                                                                                                         |
|:------| :--- | :---: | :--- | :--- |:--------------------------------------------------------------------------------------------------------------------|
| `-i`  | `--input` | Yes | `str` | — | Absolute path or filename of the input binary segmentation mask (`.nii` or `.nrrd`).                                |
| `-o`  | `--output` | Yes | `str` | — | Absolute output path or filename for the computed graph (`.pkl` is appended automatically if omitted).              |
| `-oh` | `--output_html` | No | `str` | `None` | Absolute output path or filename to save an HTML visualization of the graph. If omitted, no HTML file is generated. |
| `-cr` | `--closing_radius` | No | `int` | `2` | Whole-number radius of the ball structuring element used in morphological closing.                                  |

### Examples

**Basic run using default project directories (bare filenames):**

```bash
python graph_construct.py -i actin_mask.nii -o actin_graph
```

**Export both the graph and an HTML visualization with a custom closing radius:**

```bash
python graph_construct.py -i actin.nrrd -o actin_graph -oh actin_vis.html -cr 3
```

**Using absolute paths:**

```bash
python graph_construct.py \
  --input C:/Users/name/project/src/data/input/actin.nii \
  --output C:/Users/name/project/src/data/graphs/actin.pkl \
  --output_html C:/Users/name/project/src/data/html/actin.html \
  --closing_radius 4
```

---

## Stage 2: Graph Analysis & Arp2/3 Detection

Analyzes a precomputed graph (`.pkl`), detects Arp2/3 branch candidates, generates a 3D picks mask, and outputs summary statistics.

### Processing Steps

1. **Load Graph:** Reads the graph representation (`.pkl`).
2. **Direction & Angle Computation:** Estimates filament directions using voxels between `lower_limit` and `upper_limit` from each intersection and computes all possible pairwise angles.
3. **Arp2/3 Candidate Filtering:** Selects intersection nodes whose angles match the target signature `(70°, 110°, 180°)` within `±threshold` degrees.
4. **Mask Generation:** Writes a `.nii` or `.nrrd` mask (matching the input mask format) with a cube marking predicted Arp2/3 intersections.
5. **Plots & Statistics:** Plots distributions of intersection degrees and subfilament lengths and prints descriptive statistics (`count`, `mean`, `std`, quartiles, `min`, `max`) to the console.

### Usage

```bash
python compute_properties.py -i <input_graph> -im <input_mask> -m <output_mask> [-t <threshold>] [-ll <lower_limit>] [-ul <upper_limit>] [-r <radius>]
```

### Command-Line Arguments

| Flag | Long Flag | Required | Type | Default | Description |
| :--- | :--- |:--------:| :--- | :--- | :--- |
| `-i` | `--input` |   Yes    | `str` | — | Path or filename of the input graph (`.pkl`) to analyze. |
| `-im` | `--input_mask` |   Yes    | `str` | — | Path or filename of the original segmentation mask (`.nii` or `.nrrd`) used to build the graph. |
| `-m` | `--output_mask_filepath` |   Yes    | `str` | — | Output path or filename for the generated mask containing predicted Arp2/3 picks. |
| `-t` | `--threshold` |    No    | `int` | `20` | Angular tolerance (degrees of freedom) around the target signature `(70°, 110°, 180°)`. |
| `-ll` | `--lower_limit` |    No    | `int` | `10` | Starting voxel index along each branch used to compute direction vectors. |
| `-ul` | `--upper_limit` |    No    | `int` | `35` | Ending voxel index along each branch considered for direction vectors. |
| `-r` | `--radius` |    No    | `int` | `None` | Sphere radius (in voxels) used to mark predicted Arp2/3 intersections in the output mask. |

### Outputs

* **Predicted picks mask:** Saved to the path provided via `-m` in `.nii` format if `--input_mask` ends with `.nii`, or `.nrrd` otherwise.
* **Distribution plots:** Saved under `DEFAULT_PLOT_PATH` using the input graph filename stem:
  * `<DEFAULT_PLOT_PATH>/<graph_name>_degrees.png`
  * `<DEFAULT_PLOT_PATH>/<graph_name>_subfil_len.png`
* **Console statistics:** summary tables for intersection degrees and subfilament length (in voxels).

### Examples

**Basic run using default project directories (bare filenames):**

```bash
python compute_properties.py -i actin_graph.pkl -im actin_mask.nii -m arp23_picks.nii
```

**Custom angle tolerance, direction voxel window, and marker sphere radius:**

```bash
python compute_properties.py \
  -i actin_graph.pkl \
  -im actin.nrrd \
  -m actin_picks.nrrd \
  -t 15 -ll 8 -ul 30 -r 4
```

**Using absolute paths:**

```bash
python compute_properties.py \
  --input C:/Users/name/project/src/data/graphs/actin.pkl \
  --input_mask C:/Users/name/project/src/data/input/actin.nii \
  --output_mask_filepath C:/Users/name/project/src/data/mask/actin_arp23.nii \
  --threshold 25 \
  --radius 3
```

---

## End-to-End Example

Run both stages back-to-back on a single segmentation mask:

```bash
# 1. Build the graph and HTML visualization from a segmentation mask ('-cr 3' is an optional argument)
python graph_construct.py -i "actin_t_cell.nrrd" -o "graph_actin_t_cell" -oh "actin_t_cell.html" -cr 3

# 2. Detect Arp2/3 intersections, export the picks mask, and plot statistics ('-t 20 -ll 10 -ul 35 -r 3' are optional arguments)
python compute_properties.py -i "actin_graph.pkl" -im "actin.nrrd" -m "actin_t_cell.nrrd" -t 20 -ll 10 -ul 35 -r 3
```