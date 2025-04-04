# CryoEM Volume Evaluation Toolkit: Chamfer Distance (CD), Volumetric IoU (vIoU)


## 📁 Structure

more simple version for just 2 structures with argparse
```
evaluate_metrics.py               # Script for fixed-threshold evaluations on 4 models
evaluate_metrics_levels.py        # Script for per-volume predicted level evaluations on 4 models
metrics_utils.py                  # Core utility + metric computation functions 
evaluate_single_model_metrics.ipynb  # Notebook to evaluate results of one model vs GT
gt_metric_visualization.ipynb    # Notebook to visualize GT-to-GT comparisons as metric validation on the dataset
config.yaml                       # Config file for fixed-threshold metrics
config_levels.yaml            # Config file for predicted-threshold metrics
```

---

## 🚀 Quick Start

### 1. Install requirements
```bash
pip install numpy pandas mrcfile scipy pyyaml matplotlib
```

### 2. Fixed-threshold evaluation
Modify `config.yaml` and run:
```bash
python evaluate_metrics.py --config config.yaml
```

This will:
- Compare predicted vs ground truth volumes using **shared, fixed isosurface thresholds**
- Output a CSV file of metrics
- Generate a figure of results comparing the four models 

### 3. Predicted-threshold evaluation
First, generate per-volume thresholds using ChimeraX (adapted from https://github.com/mariacarreira/calc_level_ChimeraX/tree/main):
```bash
chimerax --nogui batch_level_predictor.py /path/to/mrc_vols
```

Make sure a `predicted_levels.txt` file is generated for each of your models, e.g.:
```
vol_001.mrc    0.0432
vol_002.mrc    0.0551
...
```

Then modify `config_levels.yaml` to point to these files, and run:
```bash
python evaluate_metrics_levels.py --config config_levels.yaml
```

---

## 📊 Notebooks

#### `evaluate_single_model_metrics.ipynb`
Evaluate one model against ground truth with a fixed set of thresholds, producing a per-threshold plot of Chamfer Distance and IoU.

#### `gt_metric_visualization.ipynb`
For **ground truth** volumes only, this notebook:
- Selects reference structures (e.g., vol_001, vol_010, ...)
- Plots their pairwise distances to all others
- Generates a heatmap showing similarity between all volumes

---

## 🛠 Utility Functions

All core logic is available in `metrics_utils.py`:
```python
from metrics_utils import load_mrc, chamfer_distance, volumetric_iou
```

These are widely-used in 3d computer vision.

