# Paper-experiment-replication
## 0 
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


## 5 Inference with allowlist


## 
