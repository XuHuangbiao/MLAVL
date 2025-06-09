# MLAVL: Language-Guided Audio-Visual Learning for Long-Term Sports Assessment

This is the code for CVPR2025 paper "Language-Guided Audio-Visual Learning for Long-Term Sports Assessment".

## Environments

- RTX 3090
- CUDA: 12.4
- Python: 3.8.19
- PyTorch: 2.4.1+cu124

## Features

The features and label files of Rhythmic Gymnastics and Fis-V dataset can be download from the [GDLT](https://github.com/xuangch/CVPR22_GDLT) repository.

The features and label files of FS1000 dataset can be download from the [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) repository.

The features and label files of FineFS dataset can be download from the [FineFS](https://github.com/yanliji/FineFS-dataset) repository.

## Running
### The following are examples only, more details coming soon!

Please fill in or select the args enclosed by {} first.

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {Ball/Clubs/Hoop/Ribbon} --lr 1e-2 --epoch {250/400/500/150} --n_decoder 2 --n_query 4 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {Ball/Clubs/Hoop/Ribbon} --n_decoder 2 --n_query 4 --dropout 0.3 --test --ckpt {the name of the used checkpoint}
```