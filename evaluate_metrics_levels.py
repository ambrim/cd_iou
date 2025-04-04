import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
from metrics_utils import load_mrc, chamfer_distance, volumetric_iou

def load_levels(threshold_file):
    levels = {}
    with open(threshold_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                filename, value = parts
                levels[filename] = float(value)
    return levels

def compute_metrics(gt_path, pred_path, threshold, mask_path=None, grid_size=128, voxel_size=4.5):
    gt_vol = load_mrc(gt_path)
    pred_vol = load_mrc(pred_path)
    mask = load_mrc(mask_path) if mask_path else None
    cd = chamfer_distance(gt_vol, pred_vol, threshold, mask, grid_size, voxel_size)
    iou = volumetric_iou(gt_vol, pred_vol, threshold, mask)
    return cd, iou

def gather_all_metrics(gt_dir, pred_models, mask_path, output_csv, grid_size, voxel_size):
    all_data = []
    for model in pred_models:
        model_name = model['name']
        pred_dir = model['path']
        levels = load_levels(model['thresholds_file'])
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "vol_*.mrc")))

        for pred_file in pred_files:
            basename = os.path.basename(pred_file)
            idx = int(basename.split('_')[-1].split('.')[0])
            gt_file = os.path.join(gt_dir, f"vol_{idx:03d}.mrc")
            if not os.path.exists(gt_file):
                print(f"Missing GT for {basename}, skipping...")
                continue
            if basename not in levels:
                print(f"Missing predicted level for {basename}, skipping...")
                continue
            threshold = levels[basename]
            cd, iou = compute_metrics(gt_file, pred_file, threshold, mask_path, grid_size, voxel_size)
            all_data.append({
                "Model": model_name,
                "Volume": basename,
                "Threshold": threshold,
                "Chamfer Distance": cd,
                "IoU": iou
            })
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
    return df

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    cfg = load_config(args.config)

    os.makedirs(cfg['output_dir'], exist_ok=True)
    output_csv = os.path.join(cfg['output_dir'], f"{cfg['dataset']}_metrics.csv")

    df = gather_all_metrics(
        cfg['gt_dir'],
        cfg['pred_models'],
        cfg.get('mask', None),
        output_csv,
        cfg.get('grid_size', 128),
        cfg.get('voxel_size', 4.5)
    )
