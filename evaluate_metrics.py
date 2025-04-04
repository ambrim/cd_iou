import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import mrcfile
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from metrics_utils import load_mrc, chamfer distance, volumetric_iou

def compute_metrics(gt_path, pred_path, thresholds, mask_path=None, grid_size=128, voxel_size=4.5):
    gt_vol = load_mrc(gt_path)
    pred_vol = load_mrc(pred_path)
    mask = load_mrc(mask_path) if mask_path else None
    results = []
    for t in thresholds:
        cd = chamfer_distance(gt_vol, pred_vol, t, mask, grid_size, voxel_size)
        iou = volumetric_iou(gt_vol, pred_vol, t, mask)
        results.append((t, cd, iou))
    return results

def gather_all_metrics(gt_dir, pred_models, thresholds, mask_path, output_csv, grid_size, voxel_size):
    all_data = []
    for model in pred_models:
        model_name = model['name']
        pred_dir = model['path']
        pred_files = sorted(glob.glob(os.path.join(pred_dir, "vol_*.mrc")))
        for pred_file in pred_files:
            basename = os.path.basename(pred_file)
            idx = int(basename.split('_')[-1].split('.')[0])
            gt_file = os.path.join(gt_dir, f"vol_{idx:03d}.mrc")
            if not os.path.exists(gt_file):
                print(f"Missing GT for {basename}, skipping...")
                continue
            metrics = compute_metrics(gt_file, pred_file, thresholds, mask_path, grid_size, voxel_size)
            for t, cd, iou in metrics:
                all_data.append({
                    "Model": model_name,
                    "Volume": basename,
                    "Threshold": t,
                    "Chamfer Distance": cd,
                    "IoU": iou
                })
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved metrics to {output_csv}")
    return df

def plot_metrics(df, dataset_name, output_dir):
    models = df['Model'].unique()
    colors = ['#da70d6', '#75E6DA', '#a569bd', '#2E5984'][:len(models)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    for i, model in enumerate(models):
        subdf = df[df['Model'] == model]
        grouped = subdf.groupby("Threshold").agg({"Chamfer Distance": ['mean', 'std'], "IoU": ['mean', 'std']}).reset_index()
        thresholds = grouped["Threshold"]
        cd_mean, cd_std = grouped["Chamfer Distance"]["mean"], grouped["Chamfer Distance"]["std"]
        iou_mean, iou_std = grouped["IoU"]["mean"], grouped["IoU"]["std"]
        axs[0].plot(thresholds, cd_mean, label=model, color=colors[i], linewidth=2)
        axs[0].fill_between(thresholds, cd_mean - cd_std, cd_mean + cd_std, alpha=0.3, color=colors[i])
        axs[1].plot(thresholds, iou_mean, label=model, color=colors[i], linewidth=2)
        axs[1].fill_between(thresholds, iou_mean - iou_std, iou_mean + iou_std, alpha=0.3, color=colors[i])

    axs[0].set_title(f"{dataset_name} - Chamfer Distance")
    axs[1].set_title(f"{dataset_name} - IoU")
    for ax in axs:
        ax.set_xlabel("Isosurface Threshold")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid(True)
        ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{dataset_name}_metrics_plot.png")
    plt.savefig(out_path, dpi=300)
    print(f"Saved plot to {out_path}")
    plt.close()

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
        cfg['thresholds'],
        cfg.get('mask', None),
        output_csv,
        cfg.get('grid_size', 128),
        cfg.get('voxel_size', 4.5)
    )
    plot_metrics(df, cfg['dataset'], cfg['output_dir'])
