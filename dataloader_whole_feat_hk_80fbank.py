import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader
from collections import defaultdict
from typing import List, Optional


# =============================================================================
# 差分计算工具（处理无填充的单条特征）
# =============================================================================
def compute_delta_2d(feat_2d: torch.Tensor, order: int = 1, window: int = 1) -> torch.Tensor:
    """
    对单条特征计算差分（支持嵌套求二阶）。
    Args:
        feat_2d: (T, D)
        order: 1 或 2
        window: 差分窗口半径
    Returns:
        delta: (T, D)，短序列返回全零张量
    """
    if feat_2d.size(0) < 3 or order == 0:
        return torch.zeros_like(feat_2d) if order > 0 else feat_2d

    # 增加批维度便于使用 pad
    feat_3d = feat_2d.unsqueeze(0)                     # (1, T, D)
    padded = F.pad(feat_3d, (0, 0, window, window), mode='replicate')
    denom = 2 * sum(n ** 2 for n in range(1, window + 1))
    delta = torch.zeros_like(feat_3d)
    T = feat_3d.size(1)
    for n in range(1, window + 1):
        delta += n * (padded[:, window + n : window + n + T, :] -
                      padded[:, window - n : window - n + T, :])
    delta = delta.squeeze(0) / denom                    # (T, D)

    if order > 1:
        delta = compute_delta_2d(delta, order - 1, window)
    return delta


# =============================================================================
# 1. 自动分桶
# =============================================================================
def build_buckets(feat_dir: str, label_dir: str, num_buckets: int = 3):
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
            T = feat.shape[1]               # 原始特征形状 (D, T)
            file_lengths.append(T)
            valid_basenames.append(fname)
        except Exception as e:
            print(f"读取特征文件失败 {fname}: {e}")
            continue

    if len(valid_basenames) == 0:
        raise RuntimeError("没有可用的样本，请检查数据目录。")

    sorted_indices = np.argsort(file_lengths)
    sorted_lengths = np.array(file_lengths)[sorted_indices]
    n_samples = len(sorted_lengths)
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

    while len(boundaries) < num_buckets:
        boundaries.append(boundaries[-1] if boundaries else 0)

    return valid_basenames, bucket_ids.tolist(), boundaries


# =============================================================================
# 2. 数据集类（输出 240 维特征）
# =============================================================================
class TIMITPhoneDataset(Dataset):
    def __init__(self,
                 feat_dir: str,
                 label_dir: str,
                 file_basenames: List[str],
                 bucket_ids: Optional[List[int]] = None):
        self.feat_dir = feat_dir
        self.label_dir = label_dir
        self.file_basenames = file_basenames
        self.bucket_ids = bucket_ids if bucket_ids is not None else [0] * len(file_basenames)

        # 缓存原始帧数（用于分桶和长度记录）
        self.lengths = []
        for fname in file_basenames:
            path = os.path.join(feat_dir, fname)
            try:
                feat = np.load(path)
                self.lengths.append(feat.shape[1])   # 原始时间维
            except Exception:
                self.lengths.append(0)

    def __len__(self):
        return len(self.file_basenames)

    def __getitem__(self, idx):
        fname = self.file_basenames[idx]
        feat_path = os.path.join(self.feat_dir, fname)
        label_path = os.path.join(self.label_dir, fname.replace('.npy', '.txt'))

        # 加载原始特征 (D, T) -> (T, D)
        feat = np.load(feat_path)
        feat_tensor = torch.as_tensor(feat.T, dtype=torch.float)      # (T, 80)

        # 计算一阶、二阶差分，均为 (T, 80)
        delta1 = compute_delta_2d(feat_tensor, order=1, window=1)
        delta2 = compute_delta_2d(feat_tensor, order=2, window=1)

        # 拼接为 (T, 240)
        feat_tensor = torch.cat([feat_tensor, delta1, delta2], dim=-1)

        # 加载标签
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

        # 长度一致性校验（极小概率出现的文件不一致问题）
        if len(label_tensor) != feat_tensor.shape[0]:
            print(f"警告: 特征/标签长度不一致 {fname}: "
                  f"feat={feat_tensor.shape[0]}, label={len(label_tensor)}，进行截断")
            min_len = min(feat_tensor.shape[0], len(label_tensor))
            feat_tensor = feat_tensor[:min_len]
            label_tensor = label_tensor[:min_len]

        return feat_tensor, label_tensor, self.lengths[idx]


# =============================================================================
# 3. 批处理函数
# =============================================================================
def collate_fn(batch):
    feats, labels, lengths = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    max_len = max(len(f) for f in feats)
    batch_size = len(feats)
    D = feats[0].shape[1]                    # 现在为 240

    padded_feat = torch.zeros(batch_size, max_len, D, dtype=feats[0].dtype)
    padded_label = torch.full((batch_size, max_len), -100, dtype=labels[0].dtype)

    for i, (f, l) in enumerate(zip(feats, labels)):
        T = len(f)
        padded_feat[i, :T] = f
        padded_label[i, :T] = l

    return padded_feat, padded_label, lengths


# =============================================================================
# 4. 分桶批次采样器
# =============================================================================
class BucketBatchSampler(Sampler):
    def __init__(self, bucket_ids: List[int], batch_size: int,
                 shuffle: bool = True, drop_last: bool = False):
        if not isinstance(bucket_ids, list):
            raise TypeError("bucket_ids must be a list")
        self.bucket_ids = bucket_ids
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

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
            start = 0
            n = len(indices)
            while start < n:
                end = start + self.batch_size
                if end > n and self.drop_last:
                    break
                batches.append(indices[start:end])
                start = end
        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        total = 0
        for indices in self.bucket_indices.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


# =============================================================================
# 5. 工厂函数
# =============================================================================
def get_timit_dataloaders(feat_dir: str,
                          label_dir: str,
                          batch_size: int = 32,
                          num_buckets: int = 3,
                          shuffle: bool = True,
                          num_workers: int = 0,
                          drop_last: bool = False):
    basenames, bucket_ids, boundaries = build_buckets(feat_dir, label_dir, num_buckets)
    print(f"共找到 {len(basenames)} 个样本")
    for i, b in enumerate(boundaries):
        print(f"  桶 {i}: 帧数 <= {b}")

    dataset = TIMITPhoneDataset(feat_dir, label_dir, basenames, bucket_ids)

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


# =============================================================================
# 6. 测试代码
# =============================================================================
if __name__ == "__main__":
    # 根据实际路径修改
    FEAT_PATH = "./data/TIMIT_80fbank_feat_and_label/TEST/feat"
    LABEL_PATH = "./data/TIMIT_80fbank_feat_and_label/TEST/label"

    train_loader = get_timit_dataloaders(FEAT_PATH, LABEL_PATH,
                                         batch_size=32,
                                         num_buckets=3)

    for batch_idx, (feats, labels, lengths) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  特征形状: {feats.shape}")   # 期望 (B, T_max, 240)
        print(f"  标签形状: {labels.shape}")  # (B, T_max)
        print(f"  原始长度: {lengths}")
        break