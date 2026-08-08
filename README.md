# PCF-AD: A Frame-level Phoneme Recognition Method Using Prior-guided Context Fusion and Allowlist Decoding on Self-Supervised Speech Representations
## Abstract
Frame‑level phoneme recognition, which produces phonetic posteriorgrams (PPGs), is a cornerstone of many speech applications. Despite the success of self‑supervised speech representations, models built on a single SSL backbone often exhibit unbalanced performance across different phoneme categories, and they struggle with confusable phonemes. Moreover, valuable linguistic priors such as word boundaries and pronunciation dictionaries are rarely exploited during decoding.To address these challenges, we propose PCF‑AD, a frame‑level phoneme recognition method that integrates Prior‑guided Context Fusion and Allowlist Decoding on self‑supervised representations. Specifically, three expert models (using Wav2Vec2, HuBERT, and WavLM features) independently produce PPGs, and a shallow Deep Feed‑forward Sequential Memory Network (DFSMN) dynamically fuses them by leveraging per‑phoneme validation accuracy as prior knowledge together with temporal context. A subsequent allowlist decoding strategy refines the fused PPG by using word‑level time alignments and a corpus‑derived allowlist dictionary: posterior probabilities of phonemes that belong to the current word are boosted, effectively injecting lexical constraints without requiring retraining.We evaluate PCF‑AD on the TIMIT and Buckeye benchmark datasets with four different acoustic backbones (DFSMN, BLSTM, TDNN, GateConv). Experimental results show that the proposed fusion strategy consistently outperforms single‑expert models and common fusion baselines, while the allowlist decoder delivers additional plug‑and‑play gains, especially for mel‑based systems. These results validate the effectiveness, robustness, and architectural generality of the proposed method.

![image](https://github.com/kk1120kk/PCF-AD/blob/main/resource/PCF-AD.jpg)
# PCF-AD: Official Implementation
PCF-AD 一种用于帧级别音素分类的方法。包含两个贡献，

C1：Prior-guided Context Fusion of the three SSL sub-model outputs

C2：Allowlist Decoding Strategy

所提出的两个贡献，应用于BLSTM，GataConv，TDNN和DFSMN4种骨干网络，在Buckeye和TIMIT上的表现：

![image](https://github.com/kk1120kk/PCF-AD/blob/main/resource/acc_in_Buckeye.jpg)
![image](https://github.com/kk1120kk/PCF-AD/blob/main/resource/acc_in_TIMIT.jpg)

## 0 工程脚本描述
使用的3个数据加载器
```
dataloader_SSL_feat.py
dataloader_whole_feat_hk_80fbank.py
expert_dataloader.py
```
"model_"开头的脚本定义了骨干模型

"train_"开头的脚本定义了各种配方的训练

"infer_"开头的脚本用于推理或者生成PPG
## 1 Prepare data
### Download the processed data
对于Buckeye和TIMIT数据集，提供了wavs,80fbnk特征,标签和帧级词序列：

帧级词序列用于allowlist decoding.

Buckeye_wavs:      [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQAHHhhC45ElQIG1W-iqpE94AdU1r3yUTcpdtJabRQKX_A0?e=LM3xXG); [BaiduNetDisk link](https://pan.baidu.com/s/1CMYpdY6ljiUXBLModFobYQ?pwd=1120)

Buckeye_80fbank:   [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQBQHAuxTCyWT4blsJXlixo9AYLKGp6CxhUV-eLDSj1kjFc?e=VKwc2U); [BaiduNetDisk link](https://pan.baidu.com/s/1nu3l9w0eFU7o4vZ98b32-g?pwd=1120)

Buckeye_label:     [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQD_AXTWKQ6GQLbx9ZzHoDgRARQoLoe3Yh7Q0xon-YGtn_g?e=XjlV7F); [BaiduNetDisk link](https://pan.baidu.com/s/1Yblovk0uWZuVWHibXc7JBw?pwd=1120)

Buckeye_word_seq:   [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQA_H_GG91vJQYtSkQyIEmZzAWFfIE8z38-VQECfQ5qzlgQ?e=Pf8SAU); [BaiduNetDisk link](https://pan.baidu.com/s/17mGPWLSOHQdXhghZsdX7-Q?pwd=1120)

TIMIT_wavs:      [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQAkS5uj_JmCS7K0PK9bWxdiAalZJWRuy-_bRI_KivXzJ6U?e=HeudbH); [BaiduNetDisk link](https://pan.baidu.com/s/1VIwN0IN6aX0SsO_jnKgsZw?pwd=1120)

TIMIT_80fbank:   [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQCBn_cACto6TI3o6JXjjDAkAZzuRaYBzkqD9S1AoK4AhrQ?e=UIFDq6); [BaiduNetDisk link](https://pan.baidu.com/s/1J0s8IBYCkjsTgAAHPAO2pg?pwd=1120)

TIMIT_label:     [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQD6w-iNbYo1Q5leFu2UYy4lAcvtAhLXs1DWWjzyY3vgoZ0?e=HPIGtC); [BaiduNetDisk link](https://pan.baidu.com/s/1BFaYr0la0Ji3XeOw0GTIyw?pwd=1120)

TIMIT_word_seq:   [OneDrive link](https://1drv.ms/u/c/c358f1155eb8fb7f/IQC-l1MKkhXdT4O5OcGSjSBfAdf_473PqNwEa3lykveM6G0?e=HoWNFT); [BaiduNetDisk link](https://pan.baidu.com/s/1iwJ9UbN204InOYcxsK5_tw?pwd=1120)

### or Use scripts to process data by yourself

### Get Wav2Vec 2.0, HuBERT, WavLM features：

首先部署这三个模型，参考[Wav2Vec 2.0](https://huggingface.co/facebook/wav2vec2-base-960h); [HuBERT](https://huggingface.co/facebook/hubert-base-ls960); [WavLM](https://huggingface.co/microsoft/wavlm-base)

./get_SSL_feat中,使用这三个脚本从wavs中提取特征：
```
w2v2_get_feat_hidden_state.py
HuBert_get_hidden_feat.py
wavlm_get_hidden_feat.py
```
对于Wav2Vec 2.0，我们使用第10层的输出作为特征， HuBERT和WavLM都提取第12层的输出作为特征。提取的特征的帧数与wav的帧数严格保持一致。
### Get 帧级词序列:
./get_Qwen3ASR_word_seq中, 使用这个脚本来获取帧级的词序列：
```
qwen3_get_WRD_ali2_use_label.py
```



## 2 Create conda env 
```
conda create -n ASR_phn python=3.10
conda activate ASR_phn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install transformers
pip install librosa

```

## 3 Train a basic acoustic model to generate PPGs(Phonetic PosteriorGram)
以DFSMN作为骨干网络,Buckeye数据集为例子。对于每个py脚本，以内置变量配置数据路径和模型参数等，需要根据具体的情况配置。

训练并推理使用mel作为特征的DFSMN，推理的时候会保存PPG。
```
python train_mel_DFSMN_Buckeye.py  
python infer_DFSMN_mel_Buckeye.py  
```
训练并推理分别使用w2v2,hubert和wavlm作为特征的三个DFSMN，推理的时候会保存PPG。
```
python train_w2v2_FSMN_Buckeye.py
python train_hubert_FSMN_Buckeye.py
python train_wavlm_FSMN_Buckeye.py

python infer_DFSMN_w2v2_get_PPG.py
python infer_DFSMN_hubert_get_PPG.py
python infer_DFSMN_wavlm_get_PPG.py
```
## 4 Train a fusion model for integrating PPGs
在训练好分别使用w2v2,hubert和wavlm作为特征的三个DFSMN后，使用Prior-guided Context Fusion模型来融合三者的PPG，需要手动配置PPG的地址。
```
python train_3SSL_expert_dfsmn_Concat_ValPriorFeat_compare.py
python infer_3SSL_expert_dfsmn_Concat_ValPriorFeat_compare.py
```
## 5 Inference with allowlist decoding
在使用allowlist解码方法前需要保存好PPG，帧级词序列，并且构建好allowlist字典.
需要配置test集的PPG路径，帧级词序列路径和allowlist字典的路径。
我们给出了构建好的字典"Buckeye3_merged_train_val_dict.txt"，"TIMIT_TRAIN_VAL_merged_phn_dict.txt"
运行脚本来使用allowlist解码方法精炼PPG，提高准确率
```
python infer_AllowlistDecode_use_PPG_compare.py
```
## 6 构建allowlist字典
运行脚本获取每个单词在语音中的所有发音变体。
请确保"./Buckeye_Corpus" 路径下
```
Buckeye_Corpus
├── s01
│   └── unzip
│       └── s0101a.words
│       └── s0101b.words
│       ...
│       └── s0103a.words
├── s02
│   └── unzip
│       └── *.words
├── s03
│   └── unzip
│       └── *.words
├── s04
│   └── unzip
│       └── *.words
│   ...
└── s40
    └── unzip
        └── *.words
```
```
python Buckeye_get_word_phn_seq_DICT.py
```
会保存发音变体字典"Buckeye3_train_vocab_with_all_prons.txt"和"Buckeye3_test_vocab_with_all_prons.txt"，格式例如：
```
across | AH K R AA S | AH K R AO S | IH K R AO S | UH K R AA S
act | AE K T | AE K
acted | AE K T AH D | AE K T IH D
acting | AE K T IH NG | AE K N
action | AE K SH AH N | AE K SH IH M
active | AE K T IH V
```

运行脚本来将TRIAN和TEST的发音变体字典构建为allowlist字典
```
python Buckeye_get_train_val_in_one_DICT.py
```



















## 
