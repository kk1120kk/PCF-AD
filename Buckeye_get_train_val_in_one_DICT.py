#!/usr/bin/env python3
"""
合并 train 和 val 的词音素集，输出为字典格式 (一行分隔符，一行排序后的音素ID)

输入文件：
  - train_vocab_with_prons.txt
  - val_vocab_with_prons.txt     (格式: 词 | 发音1 | 发音2 ...)

输出：合并后的字典文件 Buckeye_merged_train_val_dict.txt
"""

import os

# ======================== 配置 ========================
TRAIN_FILE = "Buckeye3_train_vocab_with_all_prons.txt"   # 训练集词表
VAL_FILE   = "Buckeye3_val_vocab_with_all_prons.txt"     # 验证集词表
OUTPUT_DICT = "Buckeye3_merged_train_val_dict.txt"   # 输出字典文件

# ARPA符号 -> ID 映射（与训练/推理使用的完全一致）
PHONEME_TO_ID = {
    'SIL': 0,
    'AA': 1, 'AE': 2, 'AH': 3, 'AO': 4, 'AW': 5,
    'AY': 6, 'EH': 7, 'ER': 8, 'EY': 9, 'IH': 10,
    'IY': 11, 'OW': 12, 'OY': 13, 'UH': 14, 'UW': 15,
    'B': 16, 'CH': 17, 'D': 18, 'DH': 19, 'F': 20,
    'G': 21, 'HH': 22, 'JH': 23, 'K': 24, 'L': 25,
    'M': 26, 'N': 27, 'NG': 28, 'P': 29, 'R': 30,
    'S': 31, 'SH': 32, 'T': 33, 'TH': 34, 'V': 35,
    'W': 36, 'Y': 37, 'Z': 38, 'ZH': 39
}
# =====================================================

def load_word_phonemes(filepath):
    """
    读取词表文件，返回 { word : set of phoneme IDs }
    词表格式: 词 | 发音1 | 发音2 ...
    每个发音由空格分隔的 ARPA 符号组成。
    """
    word2ids = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('|')
            word = parts[0].strip().lower()
            id_set = set()
            for pron_str in parts[1:]:
                for ph in pron_str.strip().split():
                    ph = ph.strip()
                    if ph in PHONEME_TO_ID:
                        id_set.add(PHONEME_TO_ID[ph])
                    else:
                        print(f"警告：未知音素 '{ph}' 出现在词 '{word}' 中，已忽略")
            word2ids[word] = id_set
    return word2ids

def merge_dicts(dict1, dict2):
    """合并两个字典，对重复词取音素集并集"""
    merged = {}
    for w, pset in dict1.items():
        merged[w] = pset.copy()
    for w, pset in dict2.items():
        if w in merged:
            merged[w] |= pset
        else:
            merged[w] = pset.copy()
    return merged

def main():
    if not os.path.isfile(TRAIN_FILE) or not os.path.isfile(VAL_FILE):
        print("错误：输入文件缺失")
        return

    print("加载训练集 ...")
    train_ph = load_word_phonemes(TRAIN_FILE)
    print(f"  {len(train_ph)} 个词")

    print("加载验证集 ...")
    val_ph = load_word_phonemes(VAL_FILE)
    print(f"  {len(val_ph)} 个词")

    print("合并 ...")
    merged = merge_dicts(train_ph, val_ph)
    print(f"合并后共 {len(merged)} 个词")

    # 写入字典文件
    with open(OUTPUT_DICT, 'w', encoding='utf-8') as f:
        for word in sorted(merged.keys()):
            ids = sorted(merged[word])
            f.write(f"--------------------------------- {word}\n")
            f.write(" ".join(str(i) for i in ids) + "\n")

    print(f"字典已保存至 {OUTPUT_DICT}")

if __name__ == "__main__":
    main()