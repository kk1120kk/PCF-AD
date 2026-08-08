#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDNN PPG 推理脚本
功能：加载训练好的 TDNN 模型，分别为训练/验证/测试集生成帧级后验概率 (T, 40) 并保存为 .npy
用法：
  1. 修改下方配置区的模型路径、数据集目录和模型超参数
  2. 运行脚本
注意：模型输入为 (B, D, T)，特征文件应为 (T, 768) 的 .npy 文件
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from compare_model_TDNN import TDNN    # 请确保模型文件在当前目录

# ======================== 配置区（请根据实际情况修改） ========================

# 1. 模型检查点路径（训练得到的最佳 .pth）
MODEL_CKPT = "/root/autodl-tmp/phn_ASR_2/hk_compare/tdnn_wavlm_Buckeye_20260731_200827/best_epoch22_valacc81.29_20260731_200827.pth"

# 2. 数据集特征目录（输入）与后验输出目录（独立配置）
# ----- 训练集 -----
TRAIN_FEAT_DIR   = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_hidden_states/layer_12/train/feat"
TRAIN_OUTPUT_DIR = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/TDNN_wavlm_Buckeye/train"

# ----- 验证集 -----
VAL_FEAT_DIR     = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_hidden_states/layer_12/val/feat"
VAL_OUTPUT_DIR   = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/TDNN_wavlm_Buckeye/val"

# ----- 测试集 -----
TEST_FEAT_DIR    = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_hidden_states/layer_12/test/feat" 
TEST_OUTPUT_DIR  = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/TDNN_wavlm_Buckeye/test"

# 3. 模型超参数（必须与训练时完全一致）
INPUT_DIM = 768
NUM_CLASSES = 40
HIDDEN_DIM = 1024           # TDNN 的 hidden_dim
NUM_LAYERS = 4
DROPOUT = 0.5

# 4. 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =====================================================================


def load_model(ckpt_path, device):
    """实例化模型并加载权重"""
    model = TDNN(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {ckpt_path}")
    return model


def inference_and_save(feat_dir, output_dir, model, device):
    """
    读取 feat_dir 中所有 .npy 特征文件，推理后验概率并保存到 output_dir
    特征文件形状：(T, 768)
    """
    if feat_dir is None or output_dir is None:
        print("Skipping empty directory.")
        return
    if not os.path.isdir(feat_dir):
        print(f"Error: feature directory not found - {feat_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)
    feat_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
    if not feat_files:
        print(f"Warning: No .npy files found in {feat_dir}")
        return

    with torch.no_grad():
        for idx, fname in enumerate(feat_files):
            # 加载特征
            feat = np.load(os.path.join(feat_dir, fname))          # (T, 768)
            feat_tensor = torch.from_numpy(feat).float().to(device)
            # TDNN 输入要求 (B, D, T)
            feat_tensor = feat_tensor.unsqueeze(0).transpose(1, 2)  # (1, 768, T)

            # 前向传播
            logits = model(feat_tensor)                             # (1, T, 40)  注意 TDNN 内部 transpose 为 (B, T, C)

            # 后验概率（softmax）
            probs = F.softmax(logits, dim=-1)                       # (1, T, 40)
            probs_np = probs.squeeze(0).cpu().numpy()               # (T, 40)

            # 保存
            out_path = os.path.join(output_dir, fname)
            np.save(out_path, probs_np)
            if (idx + 1) % 100 == 0 or (idx + 1) == len(feat_files):
                print(f"Processed {idx+1}/{len(feat_files)}: {fname}")


def main():
    model = load_model(MODEL_CKPT, DEVICE)

    # 构建任务列表
    tasks = [
        ('train', TRAIN_FEAT_DIR, TRAIN_OUTPUT_DIR),
        ('val',   VAL_FEAT_DIR,   VAL_OUTPUT_DIR),
        ('test',  TEST_FEAT_DIR,  TEST_OUTPUT_DIR),
    ]

    for split_name, feat_dir, out_dir in tasks:
        print(f"\n>>> Processing {split_name} set")
        print(f"    Features from : {feat_dir}")
        print(f"    Output to     : {out_dir}")
        inference_and_save(feat_dir, out_dir, model, DEVICE)

    print("\nAll done. Results saved to configured output directories.")


if __name__ == "__main__":
    main()