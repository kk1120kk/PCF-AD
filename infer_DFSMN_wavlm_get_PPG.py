#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSL 特征 + DeepFSMN 推理脚本 —— 独立路径配置版
功能：加载训练好的模型，分别对训练集、验证集、测试集的特征进行推理，
     将每帧后验概率 (T, 40) 保存为 npy 文件。
用法：
  1. 修改下方配置区的路径和模型超参数。
  2. 运行脚本即可。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from model_deep_fsmn import DeepFSMN           # 请确保模型文件在当前目录

# # ======================== 配置区（TIMIT） ========================

# # 1. 模型检查点路径（训练得到的最佳 .pth）
# MODEL_CKPT = "/root/autodl-tmp/phn_ASR_2/hk_train_ckpt/wavlm_HIDDEN=12_dfsmn_win=10_LR=1e-4_dropout=05_layer=4_20260722_155036/best_epoch28_valacc86.36_20260722_155036.pth"

# # 2. 数据集特征目录（输入）与后验输出目录（独立配置）
# #    提示：若某个数据集不需要推理，可将对应目录设为 None 或跳过

# # ----- 训练集 -----
# TRAIN_FEAT_DIR   = "/root/autodl-tmp/phn_ASR_2/data/wavlm_hidden_states/layer_12/TRAIN/feat"
# TRAIN_OUTPUT_DIR = "/root/autodl-tmp/phn_ASR_2/data/wavlm_PPG/train"

# # ----- 验证集 -----
# VAL_FEAT_DIR     = "/root/autodl-tmp/phn_ASR_2/data/wavlm_hidden_states/layer_12/VAL/feat"
# VAL_OUTPUT_DIR   = "/root/autodl-tmp/phn_ASR_2/data/wavlm_PPG/val"

# # ----- 测试集 -----
# TEST_FEAT_DIR    = "/root/autodl-tmp/phn_ASR_2/data/wavlm_hidden_states/layer_12/TEST/feat" 
# TEST_OUTPUT_DIR  = "/root/autodl-tmp/phn_ASR_2/data/wavlm_PPG/test"

# ======================== 配置区（Buckeye） ========================

# 1. 模型检查点路径（训练得到的最佳 .pth）
MODEL_CKPT = "/root/autodl-tmp/phn_ASR_2/hk_train_ckpt_Buckeye_new_split3/wavlm_HIDDEN=12_dfsmn_win=10_LR=1e-4_dropout=05_layer=4_20260730_170158/best_epoch35_valacc79.64_20260730_170158.pth"

# 2. 数据集特征目录（输入）与后验输出目录（独立配置）
#    提示：若某个数据集不需要推理，可将对应目录设为 None 或跳过

# ----- 训练集 -----
TRAIN_FEAT_DIR   = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_hidden_states/layer_12/train/feat"
TRAIN_OUTPUT_DIR = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_PPG/train"

# ----- 验证集 -----
VAL_FEAT_DIR     = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_hidden_states/layer_12/val/feat"
VAL_OUTPUT_DIR   = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_PPG/val"

# ----- 测试集 -----
TEST_FEAT_DIR    = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_hidden_states/layer_12/test/feat" 
TEST_OUTPUT_DIR  = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_PPG/test"


# 3. 模型超参数（必须与训练时完全一致）
INPUT_DIM = 768
NUM_CLASSES = 40
HIDDEN_DIM = 1024
NUM_LAYERS = 4
LEFT_CONTEXT = 10
RIGHT_CONTEXT = 10
DROPOUT = 0.5
ACTIVATION = 'relu'

# 4. 设备
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# =====================================================================


def load_model(ckpt_path, device):
    """实例化模型并加载权重"""
    model = DeepFSMN(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        left_context=LEFT_CONTEXT,
        right_context=RIGHT_CONTEXT,
        dropout=DROPOUT,
        activation=ACTIVATION
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
            # 模型输入要求 (batch, dim, time)，batch=1
            feat_tensor = feat_tensor.unsqueeze(0).transpose(1, 2)  # (1, 768, T)

            # 前向传播
            logits = model(feat_tensor)                             # (1, T, 40)

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

    # 构建任务列表，直接使用配置好的路径
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