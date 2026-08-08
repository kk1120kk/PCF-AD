"""
FSMN 专家融合模型训练脚本（拼接先验版）
将先验矩阵展平并拼接到输入中，让网络自行学习如何利用先验。
"""
import os
import datetime
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from model_3SSL_expert_dfsmn_Concat_ValPriorFeat import ExpertFusionFSMN_ConcatPrior
from expert_dataloader import get_expert_dataloader

# # ========================= 配置区(Buckeye) =========================
CONFIG = {
    # 数据路径
    "train_prob_dirs": [
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_w2v2_Buckeye/train",
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_hubert_Buckeye/train",
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_wavlm_Buckeye/train"
    ],
    "train_label_dir": "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/train",

    "val_prob_dirs": [
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_w2v2_Buckeye/val",
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_hubert_Buckeye/val",
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_wavlm_Buckeye/val"
    ],
    "val_label_dir": "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/val",

    "test_prob_dirs": [
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_w2v2_Buckeye/test",
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_hubert_Buckeye/test",
        "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_wavlm_Buckeye/test"
    ],
    "test_label_dir": "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/test",

    # ------------------ 模型超参数（FSMN）------------------
    "num_classes": 40,
    "num_experts": 3,
    "fsmn_hidden": 256,           # FSMN 隐藏层维度
    "fsmn_layers": 2,             # FSMN 层数
    "left_context": 2,           # 左侧上下文帧数
    "right_context": 2,          # 右侧上下文帧数
    "dropout": 0.3,
    "activation": "relu",

    # ------------------ 先验设置 ------------------
    "use_prior": True,            # True: 使用验证集先验并拼接到输入; False: 完全不使用先验

    # ------------------ 训练超参数 ------------------
    "batch_size": 64,
    "num_epochs": 20,
    "learning_rate": 2e-4,
    "weight_decay": 1e-3,
    "seed": 1120,
    "num_workers": 2,

    # ------------------ 保存设置 ------------------
    "save_root": "hk_compare_3SSL_Expert",
    "experiment_name": "GateConv_Buckeye_CatPrior",
    "save_freq": 5,
}

# # # ========================= 配置区(TIMIT) =========================
# CONFIG = {
#     # ------------------ 数据路径（请按实际修改）------------------
#     "train_prob_dirs": [
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_w2v2_TIMIT/train",
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_hubert_TIMIT/train",
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_wavlm_TIMIT/train"
#     ],
#     "train_label_dir": "/root/autodl-tmp/phn_ASR_2/TIMIT_data/timit_label_20ms/hk_train/label",

#     "val_prob_dirs": [
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_w2v2_TIMIT/val",
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_hubert_TIMIT/val",
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_wavlm_TIMIT/val"
#     ],
#     "val_label_dir": "/root/autodl-tmp/phn_ASR_2/TIMIT_data/timit_label_20ms/hk_val/label",

#     "test_prob_dirs": [
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_w2v2_TIMIT/test",
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_hubert_TIMIT/test",
#         "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_wavlm_TIMIT/test"
#     ],
#     "test_label_dir": "/root/autodl-tmp/phn_ASR_2/TIMIT_data/timit_label_20ms/hk_test/label",

#     # ------------------ 模型超参数（FSMN）------------------
#     "num_classes": 40,
#     "num_experts": 3,
#     "fsmn_hidden": 256,           # FSMN 隐藏层维度
#     # "fsmn_layers": 3,             # FSMN 层数
#     # "left_context": 10,           # 左侧上下文帧数
#     # "right_context": 10,          # 右侧上下文帧数
#      "fsmn_layers": 2,             # FSMN 层数
#     "left_context": 2,           # 左侧上下文帧数
#     "right_context": 2,          # 右侧上下文帧数
#     "dropout": 0.3,
#     "activation": "relu",

#     # ------------------ 先验设置 ------------------
#     "use_prior": True,            # True: 使用验证集先验并拼接到输入; False: 完全不使用先验

#     # ------------------ 训练超参数 ------------------
#     "batch_size": 64,
#     "num_epochs": 20,
#     "learning_rate": 1e-4,
#     "weight_decay": 1e-3,
#     "seed": 1120,
#     "num_workers": 2,

#     # ------------------ 保存设置 ------------------
#     "save_root": "hk_compare_3SSL_Expert",
#     "experiment_name": "GateConv_TIMIT_CatPrior",
#     "save_freq": 5,
# }

# ======================== 先验矩阵计算 ========================
def compute_prior_matrix(prob_dirs, label_dir, batch_size=64):
    from torch.utils.data import DataLoader
    from expert_dataloader import ExpertPPGDataset, collate_fn
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

# ======================== 工具函数 ========================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_log_file(log_path, config_dict, timestamp):
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("ExpertFusion FSMN (ConcatPrior) Training Log\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("Configuration:\n")
        for key, value in config_dict.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n--- Epoch Records ---\n")

def log_epoch(log_path, epoch, train_loss, val_loss, train_acc, val_acc, test_acc, epoch_time):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"epo: {epoch+1:03d}; "
                f"train_loss: {train_loss:.6f}; val_loss: {val_loss:.6f}; "
                f"train_acc: {train_acc:.4f}; val_acc: {val_acc:.4f}; "
                f"test_acc: {test_acc:.4f}; time: {epoch_time:.1f}s\n")

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="Evaluation"):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for padded_x, padded_label in tqdm(loader, desc=desc, leave=False):
        padded_x = padded_x.to(device)
        padded_label = padded_label.to(device)

        fused_probs, _ = model(padded_x)
        log_probs = torch.log(fused_probs.reshape(-1, fused_probs.shape[-1]) + 1e-7)
        loss = criterion(log_probs, padded_label.reshape(-1))

        mask = padded_label != -100
        preds = fused_probs.argmax(dim=-1)
        correct += (preds[mask] == padded_label[mask]).sum().item()
        total += mask.sum().item()
        total_loss += loss.item() * mask.sum().item()

    avg_loss = total_loss / total if total > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def train_one_epoch(epoch, model, train_loader, val_loader, test_loader, optimizer, criterion,
                    best_acc, save_path, save_freq, device, log_path, timestamp, last_best_path):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d} [Train]", leave=False)
    for padded_x, padded_label in pbar:
        padded_x = padded_x.to(device)
        padded_label = padded_label.to(device)

        fused_probs, _ = model(padded_x)
        log_probs = torch.log(fused_probs.reshape(-1, fused_probs.shape[-1]) + 1e-7)
        loss = criterion(log_probs, padded_label.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        mask = padded_label != -100
        preds = fused_probs.argmax(dim=-1)
        correct += (preds[mask] == padded_label[mask]).sum().item()
        total += mask.sum().item()
        train_loss += loss.item() * mask.sum().item()

        current_acc = correct / total if total > 0 else 0.0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2%}'})

    avg_train_loss = train_loss / total if total > 0 else 0.0
    avg_train_acc = correct / total if total > 0 else 0.0

    avg_val_loss, avg_val_acc = evaluate(model, val_loader, criterion, device, desc="Validation")
    avg_test_loss, avg_test_acc = evaluate(model, test_loader, criterion, device, desc="Testing")

    epoch_time = time.time() - start_time
    log_epoch(log_path, epoch, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, avg_test_acc, epoch_time)

    if (epoch + 1) % save_freq == 0:
        ckpt_name = f"epoch{epoch+1}_valacc{100*avg_val_acc:.2f}_testacc{100*avg_test_acc:.2f}_{timestamp}.pth"
        ckpt_path = os.path.join(save_path, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint saved: {ckpt_name}")

    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        if last_best_path and os.path.exists(last_best_path):
            os.remove(last_best_path)
            print(f"Deleted previous best model: {os.path.basename(last_best_path)}")
        best_name = f"best_val{100*avg_val_acc:.2f}_test{100*avg_test_acc:.2f}_{timestamp}.pth"
        best_path = os.path.join(save_path, best_name)
        torch.save(model.state_dict(), best_path)
        print(f"New best model saved: {best_name}")
        last_best_path = best_path

    return best_acc, last_best_path

# ======================== 主程序 ========================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_subdir = os.path.join(CONFIG["save_root"], f"{CONFIG['experiment_name']}_{timestamp}")
    os.makedirs(save_subdir, exist_ok=True)
    log_path = os.path.join(save_subdir, f"{CONFIG['experiment_name']}_{timestamp}.txt")

    set_seed(CONFIG["seed"])
    create_log_file(log_path, CONFIG, timestamp)

    # ---------- 根据配置决定是否使用先验 ----------
    if CONFIG["use_prior"]:
        print("使用验证集计算先验矩阵...")
        prior_matrix = compute_prior_matrix(
            CONFIG["val_prob_dirs"], CONFIG["val_label_dir"], CONFIG["batch_size"]
        )
        print(f"Prior shape: {prior_matrix.shape}, min: {prior_matrix.min():.4f}, max: {prior_matrix.max():.4f}")
    else:
        print("不使用先验矩阵。")
        prior_matrix = None

    # 加载数据
    print("Loading training data...")
    train_loader = get_expert_dataloader(
        CONFIG["train_prob_dirs"], CONFIG["train_label_dir"],
        batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"]
    )
    print("Loading validation data...")
    val_loader = get_expert_dataloader(
        CONFIG["val_prob_dirs"], CONFIG["val_label_dir"],
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"]
    )
    print("Loading test data...")
    test_loader = get_expert_dataloader(
        CONFIG["test_prob_dirs"], CONFIG["test_label_dir"],
        batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"]
    )

    # 模型初始化
    model = ExpertFusionFSMN_ConcatPrior(
        num_classes=CONFIG["num_classes"],
        num_experts=CONFIG["num_experts"],
        fsmn_hidden=CONFIG["fsmn_hidden"],
        fsmn_layers=CONFIG["fsmn_layers"],
        left_context=CONFIG["left_context"],
        right_context=CONFIG["right_context"],
        dropout=CONFIG["dropout"],
        activation=CONFIG["activation"],
        prior_matrix=prior_matrix      # 传入先验矩阵或 None
    ).to(device)

    criterion = nn.NLLLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=CONFIG["learning_rate"],
                                 weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    best_acc = 0.0
    last_best_path = None

    for epoch in range(CONFIG["num_epochs"]):
        best_acc, last_best_path = train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            best_acc=best_acc,
            save_path=save_subdir,
            save_freq=CONFIG["save_freq"],
            device=device,
            log_path=log_path,
            timestamp=timestamp,
            last_best_path=last_best_path
        )
        scheduler.step(best_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | LR {current_lr:.2e} | Best Val Acc {best_acc:.4f}")

    print(f"Training completed. Best validation accuracy: {best_acc:.4f}")