# Paper-experiment-replication

## Prepare data


## Create conda env 
```
conda create -n ASR_phn python=3.10
conda activate ASR_phn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install transformers
pip install librosa

```

## Train a basic acoustic model to generate PPGs(Phonetic PosteriorGram)




