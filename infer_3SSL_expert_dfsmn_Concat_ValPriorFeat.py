#!/usr/bin/env python3
"""
FSMN 拼接先验专家融合模型 — 推理脚本
生成测试集融合 PPG 并保存为 .npy 文件
"""
import os
import torch
import numpy as np
from tqdm import tqdm

from model_3SSL_expert_dfsmn_Concat_ValPriorFeat import ExpertFusionFSMN_ConcatPrior

# # ======================== 配置区 (TIMIT)========================
# # 1. 模型权重文件路径
# CHECKPOINT = "/root/autodl-tmp/phn_ASR_2/expert_ckpt3/TIMIT_FSMN_CatPrior_20260730_193126/best_val87.43_test87.41_20260730_193126.pth"

# # 2. 测试集专家概率目录（顺序：W2V2, HuBERT, WavLM）
# TEST_PROB_DIRS = [
#     "/root/autodl-tmp/phn_ASR_2/data/w2v2_PPG/test",
#     "/root/autodl-tmp/phn_ASR_2/data/hubert_PPG/test",
#     "/root/autodl-tmp/phn_ASR_2/data/wavlm_PPG/test"
# ]

# # 3. 输出目录（融合后的 PPG 保存位置）
# OUTPUT_DIR = "/root/autodl-tmp/phn_ASR_2/data/expert_PPG_FSMN_CatPrior_2Layer_4Context/test"

# VAL_PROB_DIRS = [
#     "/root/autodl-tmp/phn_ASR_2/data/w2v2_PPG/val",
#     "/root/autodl-tmp/phn_ASR_2/data/hubert_PPG/val",
#     "/root/autodl-tmp/phn_ASR_2/data/wavlm_PPG/val"
# ]
# VAL_LABEL_DIR = "/root/autodl-tmp/phn_ASR_2/data/timit_label_20ms/hk_val/label"

# # 4. 是否保存融合 .npy 文件
# SAVE_PPG = True

# # 5. 是否使用先验（必须与训练时的配置一致）
# USE_PRIOR = True

# # 6. 先验来源（仅当 USE_PRIOR = True 时有效）
# PRIOR_SOURCE = "compute"          # "compute": 从验证集自动计算； "file": 从指定 .npy 加载
# PRIOR_MATRIX_PATH = None          # 若用 "file"，填写 .npy 路径（形状 (3,40) 或 (40,3)，脚本会自动处理）
# # 若 PRIOR_SOURCE = "compute"，需要提供验证集路径（用于计算先验矩阵）

# # 7. 模型超参数（必须与训练时完全一致）
# NUM_CLASSES = 40
# NUM_EXPERTS = 3
# FSMN_HIDDEN = 256
# FSMN_LAYERS = 2
# LEFT_CTX = 2
# RIGHT_CTX = 2
# DROPOUT = 0.3
# ACTIVATION = "relu"

# ======================== 配置区 (Buckeye)========================
# 1. 模型权重文件路径
CHECKPOINT = "/root/autodl-tmp/phn_ASR_2/expert_ckpt3/Buckeye3_FSMN_CatPrior_20260730_200247/best_val80.70_test80.73_20260730_200247.pth"

# 2. 测试集专家概率目录（顺序：W2V2, HuBERT, WavLM）
TEST_PROB_DIRS = [
    "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/w2v2_PPG/test",
    "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/hubert_PPG/test",
    "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_PPG/test"
]

# 3. 输出目录（融合后的 PPG 保存位置）
OUTPUT_DIR = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/expert_PPG_FSMN_CatPrior_2Layers_4Contexts/test"

VAL_PROB_DIRS = [
    "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/w2v2_PPG/val",
    "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/hubert_PPG/val",
    "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/wavlm_PPG/val"
]
VAL_LABEL_DIR = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/val"

# 4. 是否保存融合 .npy 文件
SAVE_PPG = True

# 5. 是否使用先验（必须与训练时的配置一致）
USE_PRIOR = True

# 6. 先验来源（仅当 USE_PRIOR = True 时有效）
PRIOR_SOURCE = "compute"          # "compute": 从验证集自动计算； "file": 从指定 .npy 加载
PRIOR_MATRIX_PATH = None          # 若用 "file"，填写 .npy 路径（形状 (3,40) 或 (40,3)，脚本会自动处理）
# 若 PRIOR_SOURCE = "compute"，需要提供验证集路径（用于计算先验矩阵）

# 7. 模型超参数（必须与训练时完全一致）
NUM_CLASSES = 40
NUM_EXPERTS = 3
FSMN_HIDDEN = 256
FSMN_LAYERS = 2
LEFT_CTX = 2
RIGHT_CTX = 2
DROPOUT = 0.3
ACTIVATION = "relu"


# ======================== 准备工作 ========================
try:
    from expert_dataloader import ExpertPPGDataset, collate_fn
    from torch.utils.data import DataLoader
    DATALOADER_OK = True
except ImportError:
    DATALOADER_OK = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 先验矩阵计算 ----------
def compute_prior_matrix(prob_dirs, label_dir, batch_size=64):
    """使用验证集统计先验准确率矩阵 (3, 40)，裁剪到 [1e-4, 1.0]"""
    if not DATALOADER_OK:
        raise RuntimeError("缺少 expert_dataloader，无法自动计算先验。请改用 'file' 模式或安装依赖。")
    dataset = ExpertPPGDataset(prob_dirs, label_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)
    correct = np.zeros((3, 40), dtype=np.int64)
    total = np.zeros(40, dtype=np.int64)
    for padded_x, padded_label in tqdm(loader, desc="Computing priors"):
        B, T, D = padded_x.shape
        expert_probs = padded_x.view(B, T, 3, 40)
        preds = expert_probs.argmax(dim=-1)
        label = padded_label.numpy()
        mask = label != -100
        for b in range(B):
            for t in range(T):
                if mask[b, t]:
                    lab = label[b, t]
                    total[lab] += 1
                    for e in range(3):
                        if preds[b, t, e] == lab:
                            correct[e, lab] += 1
    total_safe = np.where(total > 0, total, 1)
    acc = correct / total_safe
    acc = np.clip(acc, 1e-4, 1.0)
    return acc

def get_prior_matrix():
    """根据配置获取先验矩阵，返回形状 (3,40) 或 None"""
    if not USE_PRIOR:
        return None
    if PRIOR_SOURCE == "compute":
        print("从验证集自动计算先验矩阵...")
        if not all(os.path.isdir(d) for d in VAL_PROB_DIRS) or not os.path.isdir(VAL_LABEL_DIR):
            raise FileNotFoundError("验证集路径无效，无法计算先验矩阵。")
        prior = compute_prior_matrix(VAL_PROB_DIRS, VAL_LABEL_DIR)
    elif PRIOR_SOURCE == "file":
        if not PRIOR_MATRIX_PATH or not os.path.exists(PRIOR_MATRIX_PATH):
            raise FileNotFoundError(f"先验矩阵文件不存在: {PRIOR_MATRIX_PATH}")
        prior = np.load(PRIOR_MATRIX_PATH)
        # 自动转置为 (3,40) 如果形状是 (40,3)
        if prior.shape == (40, 3):
            prior = prior.T
        assert prior.shape == (3, 40), f"先验矩阵形状应为 (3,40)，实际为 {prior.shape}"
    else:
        raise ValueError("PRIOR_SOURCE 必须为 'compute' 或 'file'")
    print(f"先验矩阵加载完成，形状: {prior.shape}, min={prior.min():.4f}, max={prior.max():.4f}")
    return prior

# ---------- 推理核心 ----------
@torch.no_grad()
def infer_files(model, prob_dirs, output_dir, save):
    """遍历测试文件，生成融合 PPG 并保存"""
    first_dir = prob_dirs[0]
    if not os.path.isdir(first_dir):
        raise FileNotFoundError(f"测试集目录不存在: {first_dir}")

    all_files = sorted([f for f in os.listdir(first_dir) if f.endswith('.npy')])
    if not all_files:
        print(f"警告：{first_dir} 中未找到 .npy 文件")
        return

    if save:
        os.makedirs(output_dir, exist_ok=True)
        print(f"融合结果将保存至: {output_dir}")

    for fname in tqdm(all_files, desc="Inference"):
        # 加载三个专家的 PPG，对齐最小帧数
        probs = []
        for d in prob_dirs:
            path = os.path.join(d, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"缺少专家文件: {path}")
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] != 40:
                raise ValueError(f"{fname} 形状错误: {arr.shape}")
            probs.append(arr)
        T = min(a.shape[0] for a in probs)
        probs = [a[:T] for a in probs]

        # 拼接成 (1, T, 120)
        inputs = np.concatenate(probs, axis=1)          # (T, 120)
        input_tensor = torch.from_numpy(inputs).unsqueeze(0).to(DEVICE, dtype=torch.float32)

        # 模型前向
        fused_probs, _ = model(input_tensor)            # (1, T, 40)
        probs_np = fused_probs.squeeze(0).cpu().numpy()

        if save:
            save_path = os.path.join(output_dir, fname)
            np.save(save_path, probs_np.astype(np.float32))

    if save:
        print(f"推理完成，共处理 {len(all_files)} 个文件，结果已保存至 {output_dir}")
    else:
        print(f"推理完成（SAVE_PPG=False，未保存文件）")

# --------------------- 主函数 ---------------------
def main():
    print(f"设备: {DEVICE}")
    print(f"使用先验: {USE_PRIOR}")

    # 获取先验矩阵（若需要）
    prior_matrix = get_prior_matrix()

    # 构建模型
    model = ExpertFusionFSMN_ConcatPrior(
        num_classes=NUM_CLASSES,
        num_experts=NUM_EXPERTS,
        fsmn_hidden=FSMN_HIDDEN,
        fsmn_layers=FSMN_LAYERS,
        left_context=LEFT_CTX,
        right_context=RIGHT_CTX,
        dropout=DROPOUT,
        activation=ACTIVATION,
        prior_matrix=prior_matrix
    ).to(DEVICE)

    # 加载训练好的权重
    print(f"加载模型权重: {CHECKPOINT}")
    state_dict = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载成功。")

    # 开始推理
    infer_files(model, TEST_PROB_DIRS, OUTPUT_DIR, SAVE_PPG)

if __name__ == "__main__":
    main()