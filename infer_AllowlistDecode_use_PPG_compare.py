"""
词约束的 PPG 音素识别评估脚本（PPG 维度：帧数 x 40）
新增功能：
  1. 边界扩充策略 —— 每帧参考其左右相邻帧的词音素集合
  2. ASR 错误判断策略 —— 对非静音词段进行 top-a1 符合率检查，
     若占比 <= 阈值，则将该词段视为 ASR 错误，对应帧按静音处理且隔离扩充
  3. 支持额外音素字典（DICT_PATH_ADD），可合并两个字典的音素集合

功能：
- 利用词对齐信息约束 PPG 每帧的候选音素集（词对应的音素 + 静音）
- 支持左右相邻帧的音素集合扩充（LEFT_ADD / RIGHT_ADD）
- 支持 top-a1 音素符合率判断，对错误词段帧进行特殊处理
- 统计总体帧准确率、样本准确率分布（分桶）
- 支持设置 SEM_ID_FACTOR 权重

输入：
- PPG_dir: PPG 特征 (.npy, 形状 (T, 40))
- WRD_ali_dir: 词对齐文本 (每行: 帧序 词文本)
- GT_LABEL: 真实音素标签 (每行: 帧序 音素ID)
- DICT_PATH: 单词到音素序列的主字典文件
- DICT_PATH_ADD: 额外的音素字典文件（可选，为空则仅用主字典）

使用：
修改下方配置变量后运行即可
"""

import os
import numpy as np

# ======================== 配置区 (TIMIT)========================
# PPG_dir = "/root/autodl-tmp/phn_ASR_2/data/w2v2_PPG/test"
PPG_dir = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_Expert_TIMIT/test"
WRD_ali_dir = "/root/autodl-tmp/phn_ASR_2/TIMIT_data/TIMIT_qwen3ASR_20ms_WRD_ali"
GT_LABEL = "/root/autodl-tmp/phn_ASR_2/TIMIT_data/timit_label_20ms/hk_test/label"
DICT_PATH = "/root/autodl-tmp/phn_ASR_2/TIMIT_data/hk_TRAIN_VAL_merged_phn_dict.txt"   # 主字典
DICT_PATH_ADD = None   
SEM_ID_FACTOR = 15          # 约束音素的权重因子
LEFT_ADD = 4                # 向左扩充的帧数
RIGHT_ADD = 4               # 向右扩充的帧数
A1 = 2                      # 每帧取概率最大的前 A1 个音素进行符合判断
USE_WRD_ALI_THRESHOLD = 0.3 # 词段符合帧占比 > 此阈值时视为正确词段
# =========================================================

# # ======================== 配置区 (Buckeye)========================
# PPG_dir = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/GateConv_Expert_Buckeye/test"
# WRD_ali_dir = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/Buckeye2_all_qwen3ASR_20ms_WRD_ali_newsplit"
# GT_LABEL = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/test"
# DICT_PATH = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/Buckeye3_merged_train_val_dict.txt"          # 主字典
# DICT_PATH_ADD = None     # 额外字典（若为空字符串则只使用主字典）
# SEM_ID_FACTOR = 15          # 约束音素的权重因子
# LEFT_ADD = 4                # 向左扩充的帧数
# RIGHT_ADD = 4               # 向右扩充的帧数
# A1 = 2                      # 每帧取概率最大的前 A1 个音素进行符合判断
# USE_WRD_ALI_THRESHOLD = 0.3 # 词段符合帧占比 > 此阈值时视为正确词段
# # =========================================================

# # ======================== 配置区 (SSL_Buckeye)========================
# # PPG_dir = "/root/autodl-tmp/phn_ASR_2/data/w2v2_PPG/test"
# PPG_dir = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/TDNN_mel_Buckeye/test"
# WRD_ali_dir = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/Buckeye2_all_qwen3ASR_20ms_WRD_ali_newsplit"
# GT_LABEL = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/labels_20ms/test"
# DICT_PATH = "/root/autodl-tmp/phn_ASR_2/Buckeye3_hk_clean_train_val_test_70_15_15/Buckeye3_merged_train_val_dict.txt"          # 主字典
# DICT_PATH_ADD = None     # 额外字典（若为空字符串则只使用主字典）
# SEM_ID_FACTOR = 15          # 约束音素的权重因子
# LEFT_ADD = 4                # 向左扩充的帧数
# RIGHT_ADD = 4               # 向右扩充的帧数
# A1 = 2                      # 每帧取概率最大的前 A1 个音素进行符合判断
# USE_WRD_ALI_THRESHOLD = 0.3 # 词段符合帧占比 > 此阈值时视为正确词段
# # =========================================================

# # ======================== 配置区 (SSL_TIMIT)========================
# PPG_dir = "/root/autodl-tmp/phn_ASR_2/hk_compare_get_PPG/DFSMN_mel_TIMIT/test"
# WRD_ali_dir = "/root/autodl-tmp/phn_ASR_2/TIMIT_data/TIMIT_qwen3ASR_20ms_WRD_ali"
# GT_LABEL = "/root/autodl-tmp/phn_ASR_2/TIMIT_data/timit_label_20ms/hk_test/label"
# DICT_PATH = "/root/autodl-tmp/phn_ASR_2/TIMIT_data/hk_TRAIN_VAL_merged_phn_dict.txt"   # 主字典
# DICT_PATH_ADD = None   
# SEM_ID_FACTOR = 15          # 约束音素的权重因子
# LEFT_ADD = 4                # 向左扩充的帧数
# RIGHT_ADD = 4               # 向右扩充的帧数
# A1 = 2                      # 每帧取概率最大的前 A1 个音素进行符合判断
# USE_WRD_ALI_THRESHOLD = 0.3 # 词段符合帧占比 > 此阈值时视为正确词段
# # =========================================================


# ======================== 构建格式 ========================
"""
PPG_dir中PPG的npy格式：形状（T，40），类型 float32
WRD_ali_dir中的txt格式：每行一帧，第一个数字是帧序号，隔一个空格是词
GT_LABEL中的txt格式：每行一帧，第一个数字是帧序号，隔一个空格是帧id，隔一个空格是音素符号

字典txt中的格式：（下面那一行数字是这个词的音素集，一个词可能有多种发音，每种发音的音素统计为该词的音素集）
--------------------------------- ability
3 10 11 16 18 25 33
--------------------------------- about
3 5 16 18 33
--------------------------------- above
3 16 33 35
--------------------------------- abruptly
3 10 11 16 25 29 30 33


"""

def load_dictionary(dict_path):
    """
    解析单个字典文件，返回 word -> set of phoneme IDs
    格式：
    --------------------------------- word
    3 6 16 18 33 38
    """
    word2phones = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('-----') and len(line) > 5:
            word = line.lstrip('-').strip().lower()
            if i + 1 < len(lines):
                phone_line = lines[i + 1].strip()
                phones = [int(p) for p in phone_line.split()]
                word2phones[word] = set(phones)
            i += 2
        else:
            i += 1
    return word2phones


def load_merged_dictionary(main_path, add_path):
    """
    加载主字典和额外字典，合并每个词的音素集合。
    如果 add_path 为空，则仅返回主字典。
    """
    main_dict = load_dictionary(main_path)
    if not add_path or not os.path.exists(add_path):
        print(f"额外字典路径为空或不存在，仅使用主字典: {main_path}")
        return main_dict

    add_dict = load_dictionary(add_path)
    # 合并两个字典
    merged_dict = {}
    all_words = set(main_dict.keys()) | set(add_dict.keys())
    for word in all_words:
        merged_set = main_dict.get(word, set()) | add_dict.get(word, set())
        merged_dict[word] = merged_set
    print(f"主字典词数: {len(main_dict)}, 额外字典词数: {len(add_dict)}, 合并后词数: {len(merged_dict)}")
    return merged_dict


def compute_error_frames(ppg, wrd_words, T, a1, threshold, word2phones):
    """
    根据 top-a1 音素符合率判断每帧是否为 ASR 错误帧
    返回 is_error: list of bool，长度 T
    """
    is_error = [False] * T

    # 划分连续相同词段（lower 后比较）
    i = 0
    while i < T:
        start = i
        current_word_lower = wrd_words[i].lower()
        while i < T and wrd_words[i].lower() == current_word_lower:
            i += 1
        end = i - 1  # 词段为 [start, end]

        # 静音段不判断，始终视为正确（不标记为错误）
        if current_word_lower == "sil":
            continue

        # 获取该词对应的音素集合（仅该词本身，不含0）
        phones_set = word2phones.get(current_word_lower, set())
        # 如果词不在字典中，phones_set 为空，则所有帧都不会符合，必定判错

        frame_count = end - start + 1
        correct_count = 0
        for j in range(start, end + 1):
            probs = ppg[j, :]
            # 取概率最大的 a1 个音素（索引）
            if a1 >= 40:
                top_ids = np.arange(40)
            else:
                top_ids = np.argpartition(-probs, a1 - 1)[:a1]
            # 检查是否至少有一个在词音素集合中
            if len(set(top_ids) & phones_set) > 0:
                correct_count += 1

        ratio = correct_count / frame_count
        if ratio <= threshold:
            # 整个词段标记为错误
            for j in range(start, end + 1):
                is_error[j] = True

    return is_error


def evaluate_with_factor(factor, word2phones, left_add, right_add, a1, threshold):
    # 获取测试样本文件基名列表（从 GT_LABEL 目录）
    gt_files = sorted([f for f in os.listdir(GT_LABEL) if f.endswith('.txt')])
    if not gt_files:
        print("错误：没有找到真实标签文件。")
        return

    total_correct = 0
    total_frames = 0
    sample_acc_list = []  # 每个样本的准确率

    for fname in gt_files:
        basename = os.path.splitext(fname)[0]
        ppg_path = os.path.join(PPG_dir, basename + '.npy')
        wrd_path = os.path.join(WRD_ali_dir, basename + '.txt')
        gt_path = os.path.join(GT_LABEL, fname)

        # 读取真实标签
        try:
            with open(gt_path, 'r') as f:
                gt_ids = [int(line.strip().split()[1]) for line in f if line.strip()]
        except Exception as e:
            print(f"读取真实标签失败 {fname}: {e}")
            continue

        # 读取词对齐
        try:
            with open(wrd_path, 'r') as f:
                wrd_words = [line.strip().split()[1] for line in f if line.strip()]
        except Exception as e:
            print(f"读取词对齐失败 {fname}: {e}")
            continue

        # 读取 PPG (形状 T x 40)
        try:
            ppg = np.load(ppg_path)  # (T, 40)
        except Exception as e:
            print(f"读取 PPG 失败 {fname}: {e}")
            continue

        # 对齐帧数（取三者的最小值）
        T1 = len(gt_ids)
        T2 = len(wrd_words)
        T3 = ppg.shape[0]
        if T1 != T2 or T1 != T3:
            print(f"警告：{basename} 帧数不一致 (gt={T1}, wrd={T2}, ppg={T3})，取最小帧数处理")
        T = min(T1, T2, T3)
        if T == 0:
            print(f"跳过 {basename}（帧数为0）")
            continue

        # 截断到相同长度
        gt_ids = gt_ids[:T]
        wrd_words = wrd_words[:T]
        ppg = ppg[:T, :]  # 取前 T 帧

        # --- 计算每一帧是否为 ASR 错误帧 ---
        is_error = compute_error_frames(ppg, wrd_words, T, a1, threshold, word2phones)

        sample_correct = 0
        for i in range(T):
            probs = ppg[i, :].copy()          # (40,)

            if is_error[i]:
                # 错误帧：视为静音，只允许 ID 0，因子固定为 1（即不扩大）
                allowed_ids_set = {0}
                effective_factor = 1.0
            else:
                # ---- 边界扩充：收集 [i-left_add, i+right_add] 范围内正确帧的词音素 ----
                allowed_ids_set = set()
                start = max(0, i - left_add)
                end = min(T - 1, i + right_add)
                for j in range(start, end + 1):
                    if is_error[j]:
                        continue   # 错误帧不参与扩展
                    word = wrd_words[j].lower()
                    if word == "sil":
                        allowed_ids_set.add(0)
                    else:
                        phone_set = word2phones.get(word, set())
                        allowed_ids_set.update(phone_set)
                        allowed_ids_set.add(0)
                # 保证至少有一个候选（极端情况加 0）
                if not allowed_ids_set:
                    allowed_ids_set.add(0)
                effective_factor = factor

            # 允许的音素概率乘以 effective_factor
            for pid in allowed_ids_set:
                if pid < 40:
                    probs[pid] *= effective_factor
            pred = np.argmax(probs)

            if pred == gt_ids[i]:
                sample_correct += 1

        sample_acc = sample_correct / T
        sample_acc_list.append(sample_acc)

        total_correct += sample_correct
        total_frames += T

    if total_frames == 0:
        print("没有有效帧，无法评估。")
        return

    overall_acc = total_correct / total_frames
    print(f"\nSEM_ID_FACTOR = {factor}, LEFT_ADD = {left_add}, RIGHT_ADD = {right_add}, "
          f"A1 = {a1}, THRESHOLD = {threshold}")
    print(f"Overall Accuracy: {total_correct}/{total_frames} = {overall_acc:.4f}")

    # 分桶统计
    buckets = [
        (0.9, 1.0, "0.9~1.0"),
        (0.8, 0.9, "0.8~0.9"),
        (0.7, 0.8, "0.7~0.8"),
        (0.6, 0.7, "0.6~0.7"),
        (0.5, 0.6, "0.5~0.6"),
        (0.0, 0.5, "0.0~0.5")
    ]
    total_samples = len(sample_acc_list)
    print(f"样本总数：{total_samples}")
    for low, high, name in buckets:
        if high == 1.0:
            count = sum(1 for acc in sample_acc_list if low <= acc <= high)
        else:
            count = sum(1 for acc in sample_acc_list if low <= acc < high)
        percent = count / total_samples * 100 if total_samples > 0 else 0
        print(f"acc在{name}的样本数：{count}\t占比{percent:.2f}%")


if __name__ == "__main__":
    # 加载合并后的音素字典
    word2phones = load_merged_dictionary(DICT_PATH, DICT_PATH_ADD)

    # 评估指定参数组合
    evaluate_with_factor(SEM_ID_FACTOR, word2phones,
                         LEFT_ADD, RIGHT_ADD,
                         A1, USE_WRD_ALI_THRESHOLD)

    # 如需测试多组参数，可参考下方循环：
    # for factor in [1, 10, 30]:
    #     for left, right in [(0,0), (2,2)]:
    #         for a1 in [1, 2, 3]:
    #             for th in [0.0, 0.1, 0.2]:
    #                 evaluate_with_factor(factor, word2phones, left, right, a1, th)