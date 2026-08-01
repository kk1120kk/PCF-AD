"""
专家融合数据加载器（适配 npy 格式）
- 读取三个基模型的帧级 PPG 文件（.npy, shape: (T,40)）
- 标签为 txt 文件，每行格式：帧序号 帧ID 音素符号
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

PHONEME_LIST = [
    "SIL", "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW", "B", "CH", "D", "DH",
    "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P",
    "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"
]
PHONEME_TO_ID = {ph: i for i, ph in enumerate(PHONEME_LIST)}

class ExpertPPGDataset(Dataset):
    def __init__(self, prob_dirs, label_dir):
        """
        prob_dirs: list of 3 str, 三个模型的 PPG 目录（.npy 文件）
        label_dir: str, 标签目录（.txt 文件）
        文件名（不含扩展名）需一一对应
        """
        assert len(prob_dirs) == 3
        self.prob_dirs = prob_dirs
        self.label_dir = label_dir

        # 搜索 .npy 文件
        npy_files = [f for f in os.listdir(prob_dirs[0]) if f.endswith('.npy')]
        if not npy_files:
            raise RuntimeError(f"No .npy files found in {prob_dirs[0]}")

        self.filenames = sorted([f[:-4] for f in npy_files])   # 去掉 .npy 后缀
        print(f"Found {len(self.filenames)} .npy files in {prob_dirs[0]}")

        # 检查标签文件
        missing = []
        for name in self.filenames:
            label_file = os.path.join(label_dir, f"{name}.txt")
            if not os.path.exists(label_file):
                missing.append(label_file)
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} label files, e.g.: {missing[:5]}")
        print(f"All {len(self.filenames)} label files found in {label_dir}")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        probs = []
        for d in self.prob_dirs:
            # 从 .npy 加载
            arr = np.load(os.path.join(d, f"{name}.npy"))   # (T, 40) float64 or float32
            prob = torch.from_numpy(arr).float()            # 转为 float32 tensor
            probs.append(prob)
        stacked = torch.cat(probs, dim=-1)                  # (T, 120)

        # 加载标签
        label_path = os.path.join(self.label_dir, f"{name}.txt")
        labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    labels.append(PHONEME_TO_ID[parts[-1]])
        labels = torch.tensor(labels, dtype=torch.long)     # (T,)

        # 长度校验
        assert stacked.size(0) == labels.size(0), f"Length mismatch in {name}: {stacked.size(0)} vs {labels.size(0)}"
        return stacked, labels

def collate_fn(batch):
    T_max = max(x.size(0) for x, y in batch)
    D = batch[0][0].size(1)   # 120
    B = len(batch)

    padded_x = torch.zeros(B, T_max, D)
    padded_label = torch.full((B, T_max), -100, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        T = x.size(0)
        padded_x[i, :T, :] = x
        padded_label[i, :T] = y
    return padded_x, padded_label

def get_expert_dataloader(prob_dirs, label_dir, batch_size, shuffle=True, num_workers=0, drop_last=False):
    dataset = ExpertPPGDataset(prob_dirs, label_dir)
    print(f"Total samples in dataset: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        collate_fn=collate_fn, num_workers=num_workers, drop_last=drop_last)
    return loader