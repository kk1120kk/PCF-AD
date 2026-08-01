"""
批量提取 HuBERT 指定中间层特征，自动末尾填充以保证输出帧数与 320 点分帧的标签一致。
输入：WAV_DIR 中的 .wav 文件（16kHz 单声道）
输出：OUTPUT_DIR 中的同名 .npy 文件，形状 (原始长度//320, hidden_dim)
      特征来自第 HIDDEN_LAYER 层 Transformer（1~12），或 HIDDEN_LAYER=0 提取 CNN 编码器输出

注意：本脚本假设使用 HuBERT Base 模型（hidden_dim=768，共13层隐藏状态）。
     若使用 Large 模型，请相应调整或改为动态获取维度。
"""

import os
import glob
import torch
import transformers
import librosa
import numpy as np
'''
hk_test_wav
hk_train_wav
hk_val_wav
'''
# ==================== 可编辑变量 ====================
WAV_DIR = "/root/autodl-tmp/phn_ASR_2/Buckeye_hk_clean_train_val_test/wavs_20ms/val"                # 输入 wav 目录
OUTPUT_DIR = "./Buckeye_hk_clean_train_val_test/hubert_hidden_states/layer_12/VAL/feat"  # 输出 npy 目录
MODEL_PATH = "/root/autodl-tmp/phn_ASR_2/Hubert_base"    # 本地 HuBERT 模型路径
TARGET_SR = 16000                             # HuBERT 要求采样率
HIDDEN_LAYER = 12                             # 提取的层索引: 0=CNN编码器输出, 1~12=第1~12层 Transformer
# ===================================================

def mask_from_lengths(lengths, max_len=None):
    if max_len is None:
        max_len = lengths.max()
    x = torch.arange(max_len, dtype=lengths.dtype, device=lengths.device)
    return x.unsqueeze(0) < lengths.unsqueeze(1)


def extract_single(wav_path, model, device, layer_idx):
    """提取单条音频的指定中间层特征，返回 (特征numpy数组, 帧数)"""
    audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
    original_len = audio_tensor.shape[1]

    # 期望输出帧数 = 按320点分帧的帧数
    desired_frames = original_len // 320
    if desired_frames == 0:  # 极短音频，返回零帧特征
        # 从模型配置动态获取 hidden_dim（避免硬编码）
        hidden_dim = model.config.hidden_size
        return np.empty((0, hidden_dim), dtype=np.float32), 0

    # 计算使模型输出 desired_frames 帧所需的最小输入长度
    required_len = (desired_frames - 1) * 320 + 400
    pad_len = max(0, required_len - original_len)

    # 末尾填充（填零）
    if pad_len > 0:
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_len))

    lengths = torch.tensor([required_len], dtype=torch.long, device=device)
    mask = mask_from_lengths(lengths, max_len=required_len).to(torch.long)

    with torch.no_grad():
        # 使用 attention_mask 参数（HuBERT 与标准 Transformers 一致）
        output = model(audio_tensor, attention_mask=mask, output_hidden_states=True)
        # hidden_states 是一个 tuple，长度 = num_hidden_layers + 1（CNN 编码器 + Transformer 层）
        # layer_idx 对应：0 -> CNN编码器；1~num_layers -> 第1~最后一层 Transformer
        selected_features = output.hidden_states[layer_idx]   # [1, T, hidden_dim]
        features = selected_features.squeeze(0).cpu().numpy().astype(np.float32)

    return features, features.shape[0]


def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载 HuBERT 模型（一次）
    model = transformers.HubertModel.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()

    # 检查 HIDDEN_LAYER 的有效性
    num_layers = model.config.num_hidden_layers  # 12 for base, 24 for large
    if not (0 <= HIDDEN_LAYER <= num_layers):
        raise ValueError(f"HIDDEN_LAYER 必须在 0~{num_layers} 之间，当前为 {HIDDEN_LAYER}")
    if HIDDEN_LAYER == 0:
        layer_desc = "CNN 编码器"
    else:
        layer_desc = f"第 {HIDDEN_LAYER} 层 Transformer"
    print(f"提取指定层: {layer_desc} (索引 {HIDDEN_LAYER})")

    # 查找所有 wav 文件
    wav_files = glob.glob(os.path.join(WAV_DIR, "*.wav"))
    if not wav_files:
        print(f"警告: 在 {WAV_DIR} 中未找到 .wav 文件")
        return
    print(f"共发现 {len(wav_files)} 个音频文件，开始提取...")

    # 逐条处理
    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        try:
            features, num_frames = extract_single(wav_path, model, device, HIDDEN_LAYER)
            out_path = os.path.join(OUTPUT_DIR, base + ".npy")
            np.save(out_path, features)
            print(f"✅ {base}.npy  帧数: {num_frames}")
        except Exception as e:
            print(f"❌ {base} 处理失败: {e}")

    print("所有特征提取完成。")


if __name__ == "__main__":
    main()