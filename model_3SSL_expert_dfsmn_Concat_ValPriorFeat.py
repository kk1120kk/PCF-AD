"""
专家融合模型（DFSMN 拼接先验版）
将先验矩阵展平并广播拼接到输入中，让网络自行决定如何利用先验。
输入：三个基模型的帧级概率拼接 (B, T, 120) + 可选先验拼接
输出：融合后的概率 (B, T, 40) 及专家权重 (B, T, 3)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ======================== FSMN 基础组件 ========================
class FSMNMemoryBlock(nn.Module):
    def __init__(self, dim, left_context, right_context):
        super().__init__()
        self.left = left_context
        self.right = right_context
        kernel_size = left_context + right_context + 1
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)
        nn.init.constant_(self.conv.weight, 0.0)

    def forward(self, x):
        x = x.transpose(1, 2)                    # (B, T, D) -> (B, D, T)
        x = F.pad(x, (self.left, self.right))
        x = self.conv(x)                         # (B, D, T)
        x = x.transpose(1, 2)                    # (B, D, T) -> (B, T, D)
        return x

class FSMNLayer(nn.Module):
    def __init__(self, dim, left_context, right_context, dropout=0.1, activation='relu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.memory = FSMNMemoryBlock(dim, left_context, right_context)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU() if activation == 'relu' else nn.GELU()

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.memory(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual

# ======================== 融合模型 ========================
class ExpertFusionFSMN_ConcatPrior(nn.Module):
    def __init__(self, num_classes=40, num_experts=3,
                 fsmn_hidden=256, fsmn_layers=4,
                 left_context=10, right_context=10,
                 dropout=0.1, activation='relu',
                 prior_matrix=None):              # 先验矩阵 (3,40) 或 None
        super().__init__()
        expert_dim = num_experts * num_classes         # 120
        self.num_classes = num_classes
        self.num_experts = num_experts

        # 输入维度：基础专家概率 + 可选先验
        if prior_matrix is not None:
            input_dim = expert_dim * 2                # 240
            # 注册为先验缓冲区，前向传播时直接使用 precomputed
            prior_flat = torch.from_numpy(prior_matrix).float().view(expert_dim)  # (120,)
            self.register_buffer('prior_flat', prior_flat)
        else:
            input_dim = expert_dim

        # 输入投影
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, fsmn_hidden),
            nn.Dropout(dropout)
        )

        # 堆叠 FSMN 层
        self.fsmn_layers = nn.ModuleList([
            FSMNLayer(fsmn_hidden, left_context, right_context, dropout, activation)
            for _ in range(fsmn_layers)
        ])

        # 门控分数映射
        self.gate_project = nn.Sequential(
            nn.LayerNorm(fsmn_hidden),
            nn.Linear(fsmn_hidden, num_experts)
        )

    def forward(self, x):
        """
        x: (B, T, 120)  三个专家的概率拼接
        """
        B, T, _ = x.shape

        # 如果存在先验，展平并广播到 (B, T, 120) 后拼接
        if hasattr(self, 'prior_flat'):
            # prior_flat: (120,) -> (1, 1, 120) -> (B, T, 120)
            prior_expand = self.prior_flat.view(1, 1, -1).expand(B, T, -1)
            x = torch.cat([x, prior_expand], dim=-1)   # (B, T, 240)

        # 输入投影
        h = self.input_proj(x)                       # (B, T, hidden)

        # FSMN 编码
        for layer in self.fsmn_layers:
            h = layer(h)                             # (B, T, hidden)

        # 门控分数
        gates = self.gate_project(h)                 # (B, T, 3)

        # 动态权重
        weights = torch.softmax(gates, dim=-1)        # (B, T, 3)

        # 加权融合（注意：x 中只有前 120 维是专家概率，即使拼接了先验，原始专家概率仍在 x[...,:120]？在这里我们直接用传入的原始 x（未拼接先验）来获取专家概率）
        expert_probs = x[..., :self.num_experts * self.num_classes].view(
            B, T, self.num_experts, self.num_classes)   # (B, T, 3, 40)
        fused_probs = (expert_probs * weights.unsqueeze(-1)).sum(dim=2)  # (B, T, 40)

        return fused_probs, weights