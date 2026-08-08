# Paper-experiment-replication
## 0 工程脚本描述
使用的3个数据加载器
```
dataloader_SSL_feat.py
dataloader_whole_feat_hk_80fbank.py
expert_dataloader.py
```
"model_"开头的脚本定义了骨干模型
"train_"开头的脚本定义了各种配方的训练
"infer_"开头的脚本用于推理或者生成PPg
## 1 Prepare data
### Download the processed data

### or Use scripts to process data by yourself

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
在训练好分别使用w2v2,hubert和wavlm作为特征的三个DFSMN后，使用Prior-guided Context Fusion模型来融合三者的PPG
```
python train_3SSL_expert_dfsmn_Concat_ValPriorFeat_compare.py
python infer_3SSL_expert_dfsmn_Concat_ValPriorFeat_compare.py
```
## 5 Inference with allowlist
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
