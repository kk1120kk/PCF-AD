import torch
import torch.nn as nn

class TDNN(nn.Module):
    """
    时延神经网络（TDNN）用于帧级音素分类。
    由若干层 Conv1d + BatchNorm + ReLU + Dropout 组成，
    空洞率随层数指数增长 (2^layer_idx)，保持时间维度不变。
    """
    def __init__(self, input_dim, num_classes, hidden_dim=512, num_layers=4, dropout=0.5):
        super().__init__()
        self.num_layers = num_layers

        layers = []
        in_channels = input_dim
        for i in range(num_layers):
            out_channels = hidden_dim
            dilation = 2 ** i          # 空洞率递增，扩大感受野
            padding = dilation          # 保持时间长度不变（same padding）
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=3,
                                    dilation=dilation, padding=padding))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels

        self.tdnn_layers = nn.Sequential(*layers)
        # 1x1 卷积映射到类别数
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, D, T)  特征张量
        Returns:
            logits: (B, T, C)  每帧的未归一化概率
        """
        # 逐层处理
        x = self.tdnn_layers(x)            # (B, hidden_dim, T)
        logits = self.classifier(x)        # (B, C, T)
        logits = logits.transpose(1, 2)    # (B, T, C)  与损失函数期望对齐
        return logits