import os
import sys
import numpy as np
import torch
import torch.nn as nn
import warnings
from chimerax.core.commands import run
from chimerax.map import open_map

warnings.filterwarnings("ignore")

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(1, 32, 7, 1, padding_mode='replicate'),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, 5, 1, padding_mode='replicate'),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 5, 1, padding_mode='replicate'),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, 1, padding_mode='replicate'),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(256, 384, 1, 1, padding_mode='replicate'),
            nn.ReLU(),
            nn.AvgPool3d(3, stride=2),
            nn.Dropout(0.3)
        )
        self.mlp = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(-1, 1, 64, 64, 64).to(dtype=torch.float32)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.mlp(x)

def normalize(vol):
    d_upper = np.percentile(vol, 99.999)
    d_lower = np.percentile(vol, 0.001)
    vol = np.clip(vol, d_lower, d_upper)
    vol = np.where(vol <= 0, 0, vol)
    vmin = np.min(vol[vol > 0])
    vmax = np.max(vol)
    norm = (vol - vmin) / (vmax - vmin)
    return 2 * norm - 1, vmin, vmax

def check_boxsize(vol): return vol.shape[0]

def downsample(vol):
    box = check_boxsize(vol)
    ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol)))
    s = box // 2 - 32
    e = s + 64
    ft_d = ft[s:e, s:e, s:e]
    vol_d = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(ft_d))).real
    return vol_d.astype(np.float32) * ((64 / box) ** 3)

def upsample(vol):
    box = check_boxsize(vol)
    ft = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(vol)))
    padded = np.zeros((64, 64, 64), dtype=np.complex64)
    padded[:vol.shape[0], :vol.shape[1], :vol.shape[2]] = ft
    interp = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(padded))).real
    return interp.astype(np.float32) * ((64 / box) ** 3)

def model_pred(vol, model, device):
    tensor = torch.from_numpy(vol).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()
    return (pred + 1) / 2

def calc_level(vol, model, device):
    box = check_boxsize(vol)
    if box > 64:
        _, vmin, vmax = normalize(vol)
        vol_d = downsample(vol)
        norm, _, _ = normalize(vol_d)
    elif box < 64:
        _, vmin, vmax = normalize(vol)
        vol_d = upsample(vol)
        norm, _, _ = normalize(vol_d)
    else:
        vol_d = vol
        norm, vmin, vmax = normalize(vol_d)

    pred = model_pred(norm, model, device)
    return pred * (vmax - vmin) + vmin

def run(session, directory):
    directory = os.path.abspath(directory)
    model = CNNModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    weights = torch.load(os.path.join(script_dir, "weights_5e6.pth"), map_location=device)
    model.load_state_dict(weights["model_state_dict"])

    mrc_files = sorted(f for f in os.listdir(directory) if f.endswith(".mrc"))
    output_path = os.path.join(directory, "predicted_levels.txt")

    with open(output_path, "w") as out_file:
        for f in mrc_files:
            fpath = os.path.join(directory, f)
            maps, _ = open_map(session, fpath)
            for volume in maps:
                m = volume.full_matrix()
                level = calc_level(m, model, device)
                volume.set_parameters(surface_levels=level)
                run(session, f'volume #{volume.id} style surface')
                run(session, f'volume #{volume.id} showOutlineBox false')
                session.logger.info(f"{f}: predicted level = {level:.6f}")
                out_file.write(f"{f}\t{level:.6f}\n")

