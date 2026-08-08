#!/usr/bin/env python3
"""
Qwen3-ASR 词对齐 + 逐帧标签生成脚本（基于已有标签确定帧数）

功能：
  - 读取 Buckeye 切分后的 WAV 文件（16 kHz）
  - 从 labels_20ms 目录下读取同名 .txt 文件，获取目标帧数
  - 使用 Qwen3-ASR 进行语音识别与强制对齐，生成每帧 20 ms 的词标签
  - 将结果保存到输出目录，每行格式：帧序号 词文本

配置：
  wav_dir     : 存放 .wav 文件的目录
  label_dir   : 存放帧标签文件的目录（用于获取目标帧数）
  save_ali_dir: 输出目录，保存生成的词对齐帧标签
  frame_len_ms: 帧长，默认 20 ms
  language    : 语言，默认 English
"""

import os
import glob
import wave
from collections import defaultdict
import torch
from qwen_asr import Qwen3ASRModel

# ======================== 内置配置变量 ========================
wav_dir = "/root/autodl-tmp/Qwen_ASR/Buckeye2_hk_clean_train_val_test_70_15_15/wavs_20ms/test"                                          # WAV 目录
label_dir = "/root/autodl-tmp/Qwen_ASR/Buckeye2_hk_clean_train_val_test_70_15_15/labels_20ms/test"  # 已有帧标签目录
save_ali_dir = "./Buckeye3_test_qwen3ASR_20ms_WRD_ali_newsplit"                    # 输出目录
frame_len_ms = 20                                           # 帧长（毫秒）
language = "English"                                        # 语音语言

# 模型参数
model_name = "Qwen/Qwen3-ASR-1.7B"
forced_aligner = "Qwen/Qwen3-ForcedAligner-0.6B"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype_str = "bfloat16"
max_inference_batch_size = 32
max_new_tokens = 512
# ===============================================================


def get_frame_alignment(items, num_frames, hop_samples, sample_rate):
    """
    生成每帧的词标签（多数投票）。
    Args:
        items: list of ForcedAlignItem，包含文字及起止时间
        num_frames: 总帧数（整数）
        hop_samples: 每帧对应的采样点数（整数）
        sample_rate: 采样率（整数）
    Returns:
        list of str，长度 num_frames
    """
    frame_texts = []
    for i in range(num_frames):
        # 使用整数运算避免浮点累加误差
        frame_start = i * hop_samples / sample_rate
        frame_end = (i + 1) * hop_samples / sample_rate

        text_dur = defaultdict(float)
        for item in items:
            overlap_start = max(item.start_time, frame_start)
            overlap_end = min(item.end_time, frame_end)
            if overlap_start < overlap_end:
                text_dur[item.text] += overlap_end - overlap_start

        # 若当前帧没有任何词覆盖，则标记为 SIL
        best_text = max(text_dur, key=text_dur.get) if text_dur else "SIL"
        frame_texts.append(best_text)
    return frame_texts


def process_one_file(model, wav_path, label_path, frame_len_ms, language):
    """
    处理单个 wav 文件，返回与 label 行数相同的帧词标签列表。
    """
    # 获取目标帧数 N（从已有标签文件的行数）
    try:
        with open(label_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        target_frames = len(lines)
    except Exception as e:
        print(f"读取标签文件 {label_path} 失败: {e}")
        return None

    if target_frames == 0:
        print(f"标签文件 {label_path} 为空，跳过")
        return None

    # 读取 WAV 采样率
    try:
        with wave.open(wav_path, 'rb') as wf:
            sample_rate = wf.getframerate()
    except Exception as e:
        print(f"无法读取 {wav_path}: {e}")
        return None

    if sample_rate != 16000:
        print(f"警告: {os.path.basename(wav_path)} 采样率为 {sample_rate} Hz，期望 16000 Hz，可能产生帧数偏差")

    # 计算每帧对应的采样点数（四舍五入取整）
    hop_samples = int(sample_rate * frame_len_ms / 1000 + 0.5)
    if hop_samples <= 0:
        print(f"无效的帧长或采样率: {sample_rate}")
        return None

    # 进行语音识别 + 强制对齐
    try:
        results = model.transcribe(
            audio=wav_path,
            language=language,
            return_time_stamps=True,
        )
    except Exception as e:
        print(f"转写 {wav_path} 出错: {e}")
        return None

    # 提取对齐项，若无识别结果则 items 为空
    if not results:
        items = []          # 全部标记为 SIL
    else:
        items = results[0].time_stamps.items

    # 生成目标帧数 N 的词标签
    frame_texts = get_frame_alignment(items, target_frames, hop_samples, sample_rate)

    if len(frame_texts) != target_frames:
        print(f"警告: {os.path.basename(wav_path)} 实际生成 {len(frame_texts)} 帧，期望 {target_frames} 帧")

    return frame_texts


def main():
    os.makedirs(save_ali_dir, exist_ok=True)

    # 数据类型转换
    dtype = getattr(torch, dtype_str)

    # 加载模型
    print("正在加载 Qwen3-ASR 模型...")
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=max_inference_batch_size,
        max_new_tokens=max_new_tokens,
        forced_aligner=forced_aligner,
        forced_aligner_kwargs=dict(
            dtype=dtype,
            device_map=device,
        ),
    )
    print("模型加载完成。")

    # 搜索 WAV 文件
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        print(f"在 {wav_dir} 中未找到 .wav 文件，退出。")
        return

    print(f"共发现 {len(wav_files)} 个音频文件。")

    for wav_path in wav_files:
        base = os.path.splitext(os.path.basename(wav_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        if not os.path.exists(label_path):
            print(f"警告: 缺少对应标签文件 {label_path}，跳过 {base}")
            continue

        print(f"Processing {base} ...")
        frame_texts = process_one_file(model, wav_path, label_path,
                                       frame_len_ms, language)
        if frame_texts is None:
            print(f"处理 {base} 失败，跳过。")
            continue

        # 保存词对齐标签
        out_path = os.path.join(save_ali_dir, base + ".txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(frame_texts):
                f.write(f"{i} {text}\n")
        print(f"已保存 {out_path} (共 {len(frame_texts)} 帧)")

    print("全部处理完成。")


if __name__ == "__main__":
    main()
