import torch
import torch.nn as nn

class BLSTM(nn.Module):
    """
    双向 LSTM 网络用于帧级音素分类。
    输入形状: (B, T, D) （batch_first=True）
    经多层双向 LSTM 后由线性层映射到类别数。
    """
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_layers=3, dropout=0.5):
        """
        Args:
            input_dim: 输入特征维度 (e.g., 240 = 80 fbank + Δ + Δ²)
            num_classes: 音素类别数
            hidden_dim: LSTM 隐层维度（单向）
            num_layers: 堆叠的 LSTM 层数
            dropout: LSTM 层间 dropout（当 num_layers > 1 时生效）及分类器前 dropout
        """
        super().__init__()
        # 多层双向 LSTM，batch_first=True 使得输入/输出为 (B, T, *)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # 双向输出维度为 hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) 语音特征
        Returns:
            logits: (B, T, C) 未归一化的帧级预测
        """
        lstm_out, _ = self.lstm(x)          # (B, T, 2 * hidden_dim)
        logits = self.classifier(lstm_out)  # (B, T, num_classes)
        return logits