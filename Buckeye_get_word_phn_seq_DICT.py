#!/usr/bin/env python3
"""
构建测试集词汇表（标准发音第一位，后跟实际发音变体）

- 读取测试集片段 .words 文件，通过绝对时间戳在 Buckeye 原始语料中定位每个词。
- 同时提取该词的标准发音（第二个分号字段）和实际发音（第三个分号字段）。
- 将所有发音映射为 ARPA 符号，汇总去重。
- 输出格式：词 | 标准发音 | 实际变体1 | 实际变体2 ...
"""

import os
import glob
from collections import defaultdict


# ======================== 配置区域 ========================
TEST_WORDS_DIR = "Buckeye3_hk_clean_train_val_test_70_15_15/words/train"
BUCKEYE_ROOT   = "./Buckeye_Corpus"          # Buckeye 语料库根目录（内部为 sXX/unzip/）
OUTPUT_FILE    = "Buckeye3_train_vocab_with_all_prons.txt"

# # ======================== 配置区域 ========================
# TEST_WORDS_DIR = "Buckeye3_hk_clean_train_val_test_70_15_15/words/val"
# BUCKEYE_ROOT   = "./Buckeye_Corpus"          # Buckeye 语料库根目录（内部为 sXX/unzip/）
# OUTPUT_FILE    = "Buckeye3_val_vocab_with_all_prons.txt"

# Buckeye → ARPA 映射（完整）
buckeye_to_arpa = {
    'ah': 'AH', 'ih': 'IH', 'eh': 'EH', 'ae': 'AE', 'aa': 'AA',
    'ao': 'AO', 'uh': 'UH', 'uw': 'UW',
    'iy': 'IY', 'ey': 'EY', 'ay': 'AY', 'oy': 'OY', 'aw': 'AW',
    'ow': 'OW',
    'er': 'ER',
    'p': 'P', 'b': 'B', 't': 'T', 'd': 'D', 'k': 'K', 'g': 'G',
    'ch': 'CH', 'jh': 'JH', 'f': 'F', 'v': 'V', 'th': 'TH',
    'dh': 'DH', 's': 'S', 'z': 'Z', 'sh': 'SH', 'zh': 'ZH',
    'hh': 'HH', 'm': 'M', 'n': 'N', 'ng': 'NG', 'l': 'L',
    'r': 'R', 'w': 'W', 'y': 'Y',
    'dx': 'D', 'tq': 'T', 'nx': 'N',
    'el': 'L', 'em': 'M', 'en': 'N', 'eng': 'NG',
    'ihn': 'IH', 'own': 'OW', 'ahn': 'AA', 'aen': 'AE',
    'ehn': 'EH', 'aan': 'AA', 'iyn': 'IY', 'ayn': 'AY',
    'SIL': 'SIL'
}
# ==========================================================


def load_full_words(full_words_path):
    """
    加载单个说话人的总 words 文件。
    返回：
        time_to_prons : dict, 键为 (end_time, word_lower)，
                       值为 (标准发音, 实际发音) 元组。
                       实际发音可能为空字符串。
    """
    time_to_prons = {}
    if not os.path.exists(full_words_path):
        print(f"警告：总 words 文件不存在 {full_words_path}")
        return time_to_prons

    data_started = False
    with open(full_words_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not data_started:
                if line == '#':
                    data_started = True
                continue
            if not line:
                continue
            parts = line.split(None, 2)        # 时间、颜色码、剩余内容
            if len(parts) < 3:
                continue
            try:
                end_time = float(parts[0])
            except ValueError:
                continue

            rest = parts[2].strip()
            fields = rest.split(';')
            if len(fields) < 2:                 # 至少要有词和标准发音
                continue

            word = fields[0].strip()
            if word.startswith('<'):            # 忽略边界标签
                continue

            canon_pron = fields[1].strip()       # 标准发音
            # 实际发音（如果存在）
            if len(fields) >= 3 and fields[2].strip():
                real_pron = fields[2].strip()
            else:
                real_pron = ""                   # 空字符串表示缺失

            time_to_prons[(end_time, word.lower())] = (canon_pron, real_pron)
    return time_to_prons


def extract_speaker_from_filename(fname):
    """从片段 words 文件名提取说话人 ID，例如 s0101b_8.words -> s0101b"""
    base = os.path.splitext(fname)[0]
    return base.rsplit('_', 1)[0]


def get_subdir_from_speaker(speaker):
    """根据说话人 ID 计算 Buckeye 子目录，例如 s0101b -> s01"""
    return speaker[:3]


def buckeye_pron_to_arpa(pron_str):
    """将 Buckeye 音素序列（空格分隔）转换为 ARPA 格式化字符串"""
    if not pron_str:
        return ""
    phones = pron_str.split()
    arpa_phones = [buckeye_to_arpa.get(p, 'SIL') for p in phones]
    return ' '.join(arpa_phones)


def main():
    if not os.path.isdir(TEST_WORDS_DIR):
        print(f"错误：测试 words 目录不存在 {TEST_WORDS_DIR}")
        return
    if not os.path.isdir(BUCKEYE_ROOT):
        print(f"错误：Buckeye 根目录不存在 {BUCKEYE_ROOT}")
        return

    # 词 → 标准发音（单独保存，用于固定第一位）
    word_canonical = {}                 # word -> ARPA 标准发音（取第一个遇到的；实际应只有一个）
    # 词 → 实际发音变体集合（不含与标准发音完全相同的）
    word_real_variants = defaultdict(set)

    # 缓存已加载的说话人字典
    speaker_cache = {}

    test_files = sorted(glob.glob(os.path.join(TEST_WORDS_DIR, "*.words")))
    if not test_files:
        print("未找到任何 .words 文件")
        return

    for fpath in test_files:
        fname = os.path.basename(fpath)
        speaker = extract_speaker_from_filename(fname)
        subdir = get_subdir_from_speaker(speaker)
        full_words_path = os.path.join(BUCKEYE_ROOT, subdir, "unzip", speaker + ".words")

        if speaker not in speaker_cache:
            print(f"正在加载说话人 {speaker} 的总 words 文件 ...")
            speaker_cache[speaker] = load_full_words(full_words_path)
        time_to_prons = speaker_cache[speaker]

        # 解析片段 words 文件
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:       # 词 相对起 相对止 绝对起 绝对止
                    continue
                word = parts[0]
                if word.startswith('<'):
                    continue
                try:
                    abs_end = float(parts[4])   # 绝对结束时间
                except ValueError:
                    continue

                key = (abs_end, word.lower())
                if key in time_to_prons:
                    canon, real = time_to_prons[key]
                    canon_arpa = buckeye_pron_to_arpa(canon)
                    real_arpa  = buckeye_pron_to_arpa(real)

                    # 保存标准发音（覆盖没关系，同一个词的标准发音理应是唯一的）
                    if word.lower() not in word_canonical:
                        word_canonical[word.lower()] = canon_arpa

                    # 只收集非空的实际发音，且不同于标准发音
                    if real_arpa and real_arpa != canon_arpa:
                        word_real_variants[word.lower()].add(real_arpa)
                # 如果找不到，可能时间戳有微小偏差，这里忽略

    # 写入输出文件
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for word in sorted(word_canonical.keys()):
            canon_arpa = word_canonical[word]
            variants = sorted(word_real_variants.get(word, set()))
            # 构建发音列表：标准发音放在第一位，然后是所有变体
            pron_list = [canon_arpa] + variants
            line = f"{word} | " + " | ".join(pron_list)
            f.write(line + "\n")

    print(f"词表已生成，包含 {len(word_canonical)} 个词")
    print(f"输出文件：{OUTPUT_FILE}")


if __name__ == "__main__":
    main()