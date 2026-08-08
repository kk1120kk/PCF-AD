#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GatedConvNet (mel 特征) 推理 Buckeye 测试集 → PPG
输出目录: /root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GatedConv_mel_Buckeye/
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
from compare_model_GateConv import GatedConvNet1D

def compute_delta_2d(feat_2d: torch.Tensor, order: int = 1, window: int = 1) -> torch.Tensor:
    if feat_2d.size(0) < 3 or order == 0:
        return torch.zeros_like(feat_2d) if order > 0 else feat_2d
    feat_3d = feat_2d.unsqueeze(0)
    padded = F.pad(feat_3d, (0, 0, window, window), mode='replicate')
    denom = 2 * sum(n ** 2 for n in range(1, window + 1))
    delta = torch.zeros_like(feat_3d)
    T = feat_3d.size(1)
    for n in range(1, window + 1):
        delta += n * (padded[:, window + n : window + n + T, :] -
                      padded[:, window - n : window - n + T, :])
    delta = delta.squeeze(0) / denom
    if order > 1:
        delta = compute_delta_2d(delta, order - 1, window)
    return delta

def main():
    MODEL_CKPT = "/root/autodl-tmp/phn_ASR_2/hk_compare/mel_gatedconv_Buckeye_20260806_105845/best_epoch38_valacc58.43_20260806_105845.pth"

    INPUT_DIR  = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/feat_80fbank_20ms/test"
    OUTPUT_DIR = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GatedConv_mel_Buckeye/test"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GatedConvNet1D(
        input_dim=240,
        num_classes=40,
        hidden_dim=512,
        num_layers=10,
        kernel_size_1d=7,
        kernel_size_gated=2,
        dropout=0.5
    ).to(device)
    model.load_state_dict(torch.load(MODEL_CKPT, map_location=device))
    model.eval()

    feat_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.npy')])
    print(f"Found {len(feat_files)} feature files in {INPUT_DIR}")

    with torch.no_grad():
        for idx, fname in enumerate(feat_files):
            feat = np.load(os.path.join(INPUT_DIR, fname))
            feat_tensor = torch.from_numpy(feat.T).float().to(device)
            d1 = compute_delta_2d(feat_tensor, order=1, window=1)
            d2 = compute_delta_2d(feat_tensor, order=2, window=1)
            feat_240 = torch.cat([feat_tensor, d1, d2], dim=-1)        # (T, 240)
            # GatedConvNet 输入 (B, D, T)
            input_tensor = feat_240.unsqueeze(0).transpose(1, 2)       # (1, 240, T)
            logits = model(input_tensor)                               # (1, T, 40)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            np.save(os.path.join(OUTPUT_DIR, fname), probs)
            if (idx + 1) % 100 == 0 or (idx + 1) == len(feat_files):
                print(f"Processed {idx+1}/{len(feat_files)}: {fname}")
    print("Done.")

if __name__ == "__main__":
    main()