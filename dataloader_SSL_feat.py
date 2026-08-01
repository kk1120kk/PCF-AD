"""
Wav2Vec2 特征数据加载器
- 输入：.npy 文件（形状: (帧数, 768)），同名 .txt 标签（每行: 帧序号 音素ID 音素符号）
- 输出：按桶分组的批次，特征 pad 到 (B, T_max, 768)，标签 pad 到 (B, T_max)
- 不进行差分特征计算
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader
from collections import defaultdict
from typing import List, Optional


# ==================================================
# 1. 自动分桶（根据帧数）
# ==================================================
def build_buckets(feat_dir: str, label_dir: str, num_buckets: int = 3):
    """
    扫描特征目录，通过标签文件存在性筛选有效样本，并按帧数排序分桶。
    返回:
        basenames: 有效文件名列表（.npy 文件名）
        bucket_ids: 每个样本的桶 ID（列表）
        boundaries: 每个桶的最大帧数（用于参考）
    """
    if not os.path.isdir(feat_dir):
        raise ValueError(f"特征目录不存在: {feat_dir}")

    feat_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
    if len(feat_files) == 0:
        raise RuntimeError(f"在 {feat_dir} 中未找到 .npy 文件")

    file_lengths = []
    valid_basenames = []
    for fname in feat_files:
        label_path = os.path.join(label_dir, fname.replace('.npy', '.txt'))
        if not os.path.isfile(label_path):
            print(f"警告: 缺少标签文件 {label_path}, 跳过样本 {fname}")
            continue
        try:
            feat = np.load(os.path.join(feat_dir, fname))
            T = feat.shape[0]               # 帧数在下标0（形状为 (T, 768)）
            file_lengths.append(T)
            valid_basenames.append(fname)
        except Exception as e:
            print(f"读取特征文件失败 {fname}: {e}")
            continue

    if len(valid_basenames) == 0:
        raise RuntimeError("没有可用的样本，请检查数据目录。")

    # 按帧数排序
    sorted_indices = np.argsort(file_lengths)
    sorted_lengths = np.array(file_lengths)[sorted_indices]
    n_samples = len(sorted_lengths)

    # 分桶
    bucket_size = int(np.ceil(n_samples / num_buckets))
    bucket_ids = np.zeros(n_samples, dtype=int)
    boundaries = []

    for i in range(num_buckets):
        start = i * bucket_size
        end = min((i + 1) * bucket_size, n_samples)
        if start >= n_samples:
            break
        bucket_ids[sorted_indices[start:end]] = i
        max_len_in_bucket = sorted_lengths[end - 1] if end > 0 else 0
        boundaries.append(max_len_in_bucket)

    # 确保 boundaries 长度与 num_buckets 一致
    while len(boundaries) < num_buckets:
        boundaries.append(boundaries[-1] if boundaries else 0)

    return valid_basenames, bucket_ids.tolist(), boundaries


# ==================================================
# 2. 数据集类（Wav2Vec2 768 维特征，无差分）
# ==================================================
class Wav2Vec2PhoneDataset(Dataset):
    def __init__(self,
                 feat_dir: str,
                 label_dir: str,
                 file_basenames: List[str],
                 bucket_ids: Optional[List[int]] = None):
        self.feat_dir = feat_dir
        self.label_dir = label_dir
        self.file_basenames = file_basenames
        self.bucket_ids = bucket_ids if bucket_ids is not None else [0] * len(file_basenames)

        # 缓存每条数据的原始帧数
        self.lengths = []
        for fname in file_basenames:
            feat = np.load(os.path.join(feat_dir, fname))
            self.lengths.append(feat.shape[0])   # (T, 768) 取第0维

    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, idx):
        fname = self.file_basenames[idx]
        feat_path = os.path.join(self.feat_dir, fname)
        label_path = os.path.join(self.label_dir, fname.replace('.npy', '.txt'))

        # 加载特征 (T, 768)
        feat = np.load(feat_path)
        feat_tensor = torch.as_tensor(feat, dtype=torch.float)

        # 加载标签：每行格式 "帧序号 音素ID 音素符号"
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        labels.append(int(parts[1]))
                    except ValueError:
                        continue
        label_tensor = torch.as_tensor(labels, dtype=torch.long)

        # 对齐检查（理论上已经一致）
        if len(label_tensor) != feat_tensor.shape[0]:
            print(f"警告: 特征/标签长度不一致 {fname}: "
                  f"feat={feat_tensor.shape[0]}, label={len(label_tensor)}，进行截断")
            min_len = min(feat_tensor.shape[0], len(label_tensor))
            feat_tensor = feat_tensor[:min_len]
            label_tensor = label_tensor[:min_len]

        return feat_tensor, label_tensor, self.lengths[idx]


# ==================================================
# 3. 批处理函数（填充到批次内最大长度）
# ==================================================
def collate_fn(batch):
    feats, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = max(len(f) for f in feats)
    batch_size = len(feats)
    D = feats[0].shape[1]          # 特征维度（768）

    padded_feat = torch.zeros(batch_size, max_len, D, dtype=torch.float)
    padded_label = torch.full((batch_size, max_len), fill_value=-100, dtype=torch.long)

    for i, (f, l) in enumerate(zip(feats, labels)):
        T = len(f)
        padded_feat[i, :T] = f
        padded_label[i, :T] = l

    return padded_feat, padded_label, lengths


# ==================================================
# 4. 分桶批次采样器
# ==================================================
class BucketBatchSampler(Sampler):
    def __init__(self, bucket_ids: List[int], batch_size: int,
                 shuffle: bool = True, drop_last: bool = False):
        if not isinstance(bucket_ids, list):
            raise TypeError("bucket_ids must be a list")
        self.bucket_ids = bucket_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # 将索引按桶分组
        self.bucket_indices = defaultdict(list)
        for idx, bid in enumerate(bucket_ids):
            self.bucket_indices[bid].append(idx)
        self.bucket_indices = dict(self.bucket_indices)

    def __iter__(self):
        batches = []
        for bid, indices in self.bucket_indices.items():
            indices = list(indices)
            if self.shuffle:
                np.random.shuffle(indices)
            # 按 batch_size 切分
            for start in range(0, len(indices), self.batch_size):
                end = start + self.batch_size
                if end > len(indices) and self.drop_last:
                    break
                batches.append(indices[start:end])
        if self.shuffle:
            np.random.shuffle(batches)
        yield from batches

    def __len__(self):
        total = 0
        for indices in self.bucket_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


# ==================================================
# 5. 工厂函数（一键构建 DataLoader）
# ==================================================
def get_w2v2_dataloader(feat_dir: str,
                        label_dir: str,
                        batch_size: int = 32,
                        num_buckets: int = 3,
                        shuffle: bool = True,
                        num_workers: int = 0,
                        drop_last: bool = False):
    basenames, bucket_ids, boundaries = build_buckets(feat_dir, label_dir, num_buckets)
    print(f"特征目录: {feat_dir}")
    print(f"标签目录: {label_dir}")
    print(f"共找到 {len(basenames)} 个有效样本")
    for i, b in enumerate(boundaries):
        print(f"  桶 {i}: 帧数 <= {b}")

    dataset = Wav2Vec2PhoneDataset(feat_dir, label_dir, basenames, bucket_ids)

    batch_sampler = BucketBatchSampler(
        bucket_ids=bucket_ids,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last
    )

    loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


# ==================================================
# 6. 测试代码（直接运行此脚本）
# ==================================================
if __name__ == "__main__":
    # ------------------ 可编辑变量 ------------------
    TRAIN_FEAT_DIR = "./data/w2v2_data/TRAIN/feat"
    TRAIN_LABEL_DIR = "./data/w2v2_data/TRAIN/label"
    TEST_FEAT_DIR = "./data/w2v2_data/TEST/feat"
    TEST_LABEL_DIR = "./data/w2v2_data/TEST/label"
    # ------------------------------------------------

    # 构建训练集 DataLoader
    print("===== 训练集 =====")
    train_loader = get_w2v2_dataloader(
        TRAIN_FEAT_DIR, TRAIN_LABEL_DIR,
        batch_size=32,
        num_buckets=3,
        shuffle=True
    )

    # 查看一个批次
    for batch_idx, (feats, labels, lengths) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  特征形状: {feats.shape}")   # (B, T_max, 768)
        print(f"  标签形状: {labels.shape}")  # (B, T_max)
        print(f"  原始长度: {lengths}")
        break

    # 构建测试集 DataLoader
    print("\n===== 测试集 =====")
    test_loader = get_w2v2_dataloader(
        TEST_FEAT_DIR, TEST_LABEL_DIR,
        batch_size=32,
        num_buckets=3,
        shuffle=False
    )
    for batch_idx, (feats, labels, lengths) in enumerate(test_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  特征形状: {feats.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  原始长度: {lengths}")
        break