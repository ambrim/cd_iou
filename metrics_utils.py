import numpy as np
import mrcfile
from scipy.spatial import cKDTree

def load_mrc(file_path):
    with mrcfile.open(file_path, permissive=True) as mrc:
        return np.array(mrc.data, dtype=np.float16)

def apply_mask(volume, mask):
    return volume * mask if mask is not None else volume

def get_voxel_coords(volume, threshold, grid_size=128, voxel_size=4.5):
    coords = np.argwhere(volume >= threshold)
    return (coords - np.array([grid_size // 2]*3)) * voxel_size

def chamfer_distance(gt_vol, pred_vol, pred_thresh, mask=None, grid_size=128, voxel_size=4.5):
    gt_vol = apply_mask(gt_vol, mask)
    pred_vol = apply_mask(pred_vol, mask)
    gt_coords = get_voxel_coords(gt_vol, 0.05, grid_size, voxel_size)
    pred_coords = get_voxel_coords(pred_vol, pred_thresh, grid_size, voxel_size)
    if gt_coords.size == 0 or pred_coords.size == 0:
        return np.nan
    gt_tree, pred_tree = cKDTree(gt_coords), cKDTree(pred_coords)
    d1, _ = pred_tree.query(gt_coords)
    d2, _ = gt_tree.query(pred_coords)
    return np.mean(d1) + np.mean(d2)

def volumetric_iou(gt_vol, pred_vol, pred_thresh, mask=None):
    gt_vol = apply_mask(gt_vol, mask)
    pred_vol = apply_mask(pred_vol, mask)
    gt_binary = (gt_vol >= 0.05)
    pred_binary = (pred_vol >= pred_thresh)
    inter = np.logical_and(gt_binary, pred_binary).sum()
    union = np.logical_or(gt_binary, pred_binary).sum()
    return inter / union if union > 0 else 0
