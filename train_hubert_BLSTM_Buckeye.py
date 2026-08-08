#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HuBERT + BLSTM 训练脚本 (自动生成)
"""
import os
import datetime
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from dataloader_w2v2feat import get_w2v2_dataloader
from compare_model_BLSTM import BLSTM

# ======================== 全局配置 ========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {device}')

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
prename = "blstm_hubert_Buckeye"

SAVE_ROOT = "hk_compare"
SAVE_SUBDIR = os.path.join(SAVE_ROOT, f"{prename}_{timestamp}")
os.makedirs(SAVE_SUBDIR, exist_ok=True)

LOG_PATH = os.path.join(SAVE_SUBDIR, f"{prename}_{timestamp}.txt")
SAVE_FREQ = 20
last_best_path = None


# ======================== 1. 日志函数 ========================
def create_log_file(log_path, config_dict):
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("HuBERT + BLSTM - Training Log\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("Configuration:\n")
        for key, value in config_dict.items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
        f.write(f"{'Epoch':>5s}  {'TrainLoss':>10s}  {'ValLoss':>10s}  "
                f"{'TrainAcc':>8s}  {'ValAcc':>8s}  {'TestAcc':>8s}  {'Time(s)':>8s}\n")
        f.write("-" * 72 + "\n")


def log_epoch(log_path, epoch, train_loss, val_loss, train_acc, val_acc, test_acc, epoch_time):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{epoch+1:5d}  {train_loss:10.6f}  {val_loss:10.6f}  "
                f"{train_acc:8.4f}  {val_acc:8.4f}  {test_acc:8.4f}  {epoch_time:8.1f}\n")


# ======================== 2. 评估函数 ========================
@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_frames = 0

    for padded_feat, padded_label, lengths in tqdm(data_loader, desc="Evaluating", leave=False):
        padded_feat = padded_feat.to(device)
        padded_label = padded_label.to(device)

        feats = padded_feat

        logits = model(feats)

        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            padded_label.reshape(-1)
        )

        mask = padded_label != -100
        preds = logits.argmax(dim=-1)
        n_valid = mask.sum().item()
        correct += (preds[mask] == padded_label[mask]).sum().item()
        total_frames += n_valid
        total_loss += loss.item() * n_valid

    avg_loss = total_loss / total_frames if total_frames > 0 else 0.0
    acc = correct / total_frames if total_frames > 0 else 0.0
    return avg_loss, acc


# ======================== 3. 训练一个 epoch ========================
def train_one_epoch(epoch, model, train_loader, val_loader, test_loader, optimizer, criterion,
                    best_acc, save_path, save_freq, device, log_path, config_dict):
    global last_best_path

    rng = np.random.RandomState(seed=config_dict['seed'] + epoch * 1000)
    aug_factor = float(rng.uniform(0.5, 1.0))

    model.train()
    train_loss = 0.0
    correct = 0
    total_frames = 0
    start_time = time.time()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for padded_feat, padded_label, lengths in pbar:
        padded_feat = padded_feat.to(device)
        padded_label = padded_label.to(device)

        # 幅度增强
        padded_feat = padded_feat * aug_factor

        feats = padded_feat

        logits = model(feats)

        loss = criterion(
            logits.reshape(-1, logits.shape[-1]),
            padded_label.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        mask = padded_label != -100
        preds = logits.argmax(dim=-1)
        n_valid = mask.sum().item()
        correct += (preds[mask] == padded_label[mask]).sum().item()
        total_frames += n_valid
        train_loss += loss.item() * n_valid

        current_acc = correct / total_frames if total_frames > 0 else 0.0
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{current_acc:.2%}'})

    avg_train_loss = train_loss / total_frames if total_frames > 0 else 0.0
    avg_train_acc = correct / total_frames if total_frames > 0 else 0.0

    # 验证 & 测试
    avg_val_loss, avg_val_acc = evaluate(model, val_loader, criterion, device)
    avg_test_loss, avg_test_acc = evaluate(model, test_loader, criterion, device)

    epoch_time = time.time() - start_time

    # 记录日志
    log_epoch(log_path, epoch, avg_train_loss, avg_val_loss,
              avg_train_acc, avg_val_acc, avg_test_acc, epoch_time)

    # 定期保存
    if (epoch + 1) % save_freq == 0:
        ckpt_name = f"epoch{epoch+1}_valacc{100*avg_val_acc:.2f}_{timestamp}.pth"
        torch.save(model.state_dict(), os.path.join(save_path, ckpt_name))
        print(f"Checkpoint saved: {ckpt_name}")

    # 最佳模型保存
    if avg_val_acc > best_acc:
        best_acc = avg_val_acc
        if last_best_path and os.path.exists(last_best_path):
            os.remove(last_best_path)
        best_name = f"best_epoch{epoch+1}_valacc{100*best_acc:.2f}_{timestamp}.pth"
        best_path = os.path.join(save_path, best_name)
        torch.save(model.state_dict(), best_path)
        print(f"New best model saved: {best_name}")
        last_best_path = best_path

    return best_acc, avg_train_loss, avg_val_loss, avg_train_acc, avg_val_acc, avg_test_acc


# ======================== 4. 主程序 ========================
def main():
    config = {
        'train_feat_dir': "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/hubert_hidden_states/layer_12/train/feat",
        'train_label_dir': "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/train",
        'val_feat_dir': "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/hubert_hidden_states/layer_12/val/feat",
        'val_label_dir': "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/val",
        'test_feat_dir': "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/hubert_hidden_states/layer_12/test/feat",
        'test_label_dir': "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/test",
        'input_dim': 768,
        'num_classes': 40,
        'hidden_dim': 512,
        'num_layers': 3,
        'dropout': 0.5,

        'batch_size': 32,
        'num_epochs': 40,
        'learning_rate': 0.0001,
        'weight_decay': 0.001,
        'seed': 1120,
        'num_workers': 2
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    create_log_file(LOG_PATH, config)

    # ---------- 数据加载 ----------
    print("Loading training data...")
    train_loader = get_w2v2_dataloader(
        feat_dir=config['train_feat_dir'],
        label_dir=config['train_label_dir'],
        batch_size=config['batch_size'],
        num_buckets=3, shuffle=True,
        num_workers=config['num_workers'], drop_last=False
    )

    print("Loading validation data...")
    val_loader = get_w2v2_dataloader(
        feat_dir=config['val_feat_dir'],
        label_dir=config['val_label_dir'],
        batch_size=config['batch_size'],
        num_buckets=3, shuffle=False,
        num_workers=config['num_workers'], drop_last=False
    )

    print("Loading test data...")
    test_loader = get_w2v2_dataloader(
        feat_dir=config['test_feat_dir'],
        label_dir=config['test_label_dir'],
        batch_size=config['batch_size'],
        num_buckets=3, shuffle=False,
        num_workers=config['num_workers'], drop_last=False
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, "
          f"Test batches: {len(test_loader)}")

    # ---------- 模型、损失、优化器 ----------
    model = BLSTM(input_dim=config['input_dim'], num_classes=config['num_classes'], hidden_dim=config['hidden_dim'], num_layers=config['num_layers'], dropout=config['dropout']).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    if config.get('pretrain_ckpt') and os.path.exists(config['pretrain_ckpt']):
        model.load_state_dict(torch.load(config['pretrain_ckpt'], map_location=device))
        print(f"Loaded pretrained weights from {config['pretrain_ckpt']}")
    else:
        print("Training from scratch.")

    # ---------- 训练循环 ----------
    best_acc = 0.0
    for epoch in range(config['num_epochs']):
        best_acc, train_loss, val_loss, train_acc, val_acc, test_acc = train_one_epoch(
            epoch=epoch,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            best_acc=best_acc,
            save_path=SAVE_SUBDIR,
            save_freq=SAVE_FREQ,
            device=device,
            log_path=LOG_PATH,
            config_dict=config
        )
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1:03d}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
              f"Test Acc: {test_acc:.2%} | Best Val Acc: {best_acc:.2%}")

    print("Training completed.")
    print(f"Best validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
