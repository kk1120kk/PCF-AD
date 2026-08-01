import torch
import torch.nn as nn
import torch.nn.functional as F


class FSMNMemoryBlock(nn.Module):
    """
    记忆模块：用一维逐通道卷积实现时延聚合
    输入形状：(B, T, D)  输出形状：(B, T, D)
    """
    def __init__(self, dim, left_context, right_context):
        """
        Args:
            dim:           特征维度（通道数）
            left_context:  左侧上下文帧数
            right_context: 右侧上下文帧数
        """
        super().__init__()
        self.left = left_context
        self.right = right_context
        kernel_size = left_context + right_context + 1
        # 逐通道卷积（groups=dim），每个特征维度独立的时延系数
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            groups=dim,
            bias=False
        )
        # 初始化为 0，使训练初期记忆模块接近恒等映射
        nn.init.constant_(self.conv.weight, 0.0)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        # 手动填充以保持序列长度不变
        x = F.pad(x, (self.left, self.right))
        x = self.conv(x)            # (B, D, T)
        # 转回 (B, T, D)
        x = x.transpose(1, 2)
        return x


class FSMNLayer(nn.Module):
    """
    一个带 Pre‑LN 残差的 FSMN 层
    结构：LN -> FC1 -> 激活 -> Dropout -> 记忆块 -> FC2 -> Dropout -> 残差相加
    """
    def __init__(self, dim, left_context, right_context, dropout=0.15, activation='relu'):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.memory = FSMNMemoryBlock(dim, left_context, right_context)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        # x: (B, T, D)
        residual = x
        # Pre‑LN
        x = self.norm(x)
        # 第一个前馈（作用在特征维，不改变时间信息）
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        # 记忆模块（时间维聚合）
        x = self.memory(x)
        # 第二个前馈
        x = self.fc2(x)
        x = self.dropout(x)
        # 残差连接
        x = x + residual
        return x


class DeepFSMN(nn.Module):
    """
    DeepFSMN 帧级音素分类器
    输入：(B, D, T)  →  输出：(B, T, C)
    """
    def __init__(self,
                 input_dim,          # 输入特征维度（如 40 维 FBank）
                 num_classes,        # 音素类别数
                 hidden_dim=512,     # 隐藏层维度
                 num_layers=6,       # FSMN 层数
                 left_context=20,    # 左侧上下文帧数
                 right_context=20,   # 右侧上下文帧数
                 dropout=0.15,
                 activation='relu'):
        super().__init__()
        # 输入投影：将原始特征维度映射到 hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout)
        )

        # 堆叠多个 FSMN 层
        self.fsmn_layers = nn.ModuleList([
            FSMNLayer(hidden_dim, left_context, right_context, dropout, activation)
            for _ in range(num_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes)
        )

        # 记录一些后处理可能用到的参数
        self.hidden_dim = hidden_dim

    
    def forward(self, x):
        """
        Args:
            x: (B, D, T)  特征张量，D 为特征维度，T 为帧数
        Returns:
            logits: (B, T, num_classes)  每帧的未归一化概率
        """
        # 转置为 (B, T, D) 便于 Linear 操作
        x = x.transpose(1, 2)           # (B, T, D)
        # 输入投影
        x = self.input_proj(x)          # (B, T, hidden_dim)
        # 逐层传播
        for layer in self.fsmn_layers:
            x = layer(x)
        # 分类头（仍然保持 (B, T, hidden_dim) -> (B, T, num_classes)）
        logits = self.classifier(x)     # (B, T, C)
        return logits


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 模拟参数
    B = 8          # batch size
    D = 40         # 输入特征维度（如 FBank）
    T = 100        # 帧数
    C = 128        # 音素类别数

    model = DeepFSMN(
        input_dim=D,
        num_classes=C,
        hidden_dim=256,
        num_layers=4,
        left_context=15,
        right_context=15,
        dropout=0.1
    )

    # 伪造输入和标签
    x = torch.randn(B, D, T)          # (batch, 特征, 帧)
    y = torch.randint(0, C, (B, T))   # 每帧一个音素标签

    # 前向传播
    logits = model(x)                 # (B, T, C)

    # 计算交叉熵损失（将所有帧展平）
    loss = F.cross_entropy(
        logits.reshape(-1, C),
        y.reshape(-1)
    )
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"损失值: {loss.item():.4f}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")