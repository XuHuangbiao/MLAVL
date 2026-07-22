# MLAVL: Language-Guided Audio-Visual Learning for Long-Term Sports Assessment

This is the code for CVPR2025 paper "Language-Guided Audio-Visual Learning for Long-Term Sports Assessment".

## Abstract
Long-term sports assessment is a challenging task in video understanding since it requires judging complex movement variations and action-music coordination. However, there is no direct correlation between the diverse background music and movements in sporting events. Previous works require a large number of model parameters to learn potential associations between actions and music. To address this issue, we propose a language-guided audio-visual learning (MLAVL) framework that models" audio-action-visual" correlations guided by low-cost language modality. In our framework, multidimensional domain-based actions form action knowledge graphs, motivating audio-visual modalities to focus on task-relevant actions. We further design a shared-specific context encoder to integrate deep multimodal semantics, and an audio-visual cross-modal fusion module to evaluate action-music consistency. To match the sport's rules, we then propose a dual-branch prompt-guided grading module to weigh both visual and audio-visual performance. Extensive experiments demonstrate that our approach achieves state-of-the-art on four public long-term sports benchmarks while maintaining low parameters.

![Framework](Framework.png)

## Environments

- RTX 3090
- CUDA: 12.4
- Python: 3.8.19
- PyTorch: 2.4.1+cu124

## Features

The features (RGB, Audio, Flow) and label files of Rhythmic Gymnastics and Fis-V dataset can be download from the [PAMFN](https://github.com/qinghuannn/PAMFN) repository.

The features (RGB, Audio) and label files of FS1000 dataset can be download from the [Skating-Mixer](https://github.com/AndyFrancesco29/Audio-Visual-Figure-Skating) repository.

The features (RGB) and label files of LOGO dataset can be download from the [UIL-AQA](https://github.com/dx199771/Interpretability-AQA/tree/main) repository.

## Pretrained Model
If you wish to extract your own action text labels, please download the [ViFi-CLIP](https://github.com/muzairkhattak/ViFi-CLIP) pretrained model and place it in:
```
weights/k400_clip_complete_finetuned_30_epochs.pth
```

## Running
### The following are examples only, more details coming soon!

**Note:** Both the FS1000 and Fis-V datasets feature figure skating scenes and share the same action text features: “action-label_FS.csv” and “FS_text_feature.npy”.

Please fill in or select the args enclosed by {} first. For example, on the **FS1000** dataset:

- Training

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --model-name {the name used to save model and log} --action-type {Ball/Clubs/Hoop/Ribbon} --lr 1e-2 --epoch {250/400/500/150} --in_dim 768 --n_head 8 --n_encoder 1 --n_decoder 3 --n_query 6 --alpha 1.0 --margin 1.0 --lr-decay cos --decay-rate 0.01 --dropout 0.3
```

- Testing

```
CUDA_VISIBLE_DEVICES={device ID} python main.py --video-path {path of video features} --audio-path {path of audio features} --train-label-path {path of label file of training set} --test-label-path {path of label file of test set} --action-type {Ball/Clubs/Hoop/Ribbon} --in_dim 768 --n_head 8 --n_encoder 1 --n_decoder 3 --n_query 6 --test --ckpt {the name of the used checkpoint}
```

Additionally, you can select a result that balances SRCC and MSE based on the specific requirements of your application.

Be patient and persistent in tuning the code to achieve new state-of-the-art results.

## Citation
If our project is helpful for your research, please consider citing:
```
@InProceedings{Xu_2025_CVPR,
    author    = {Xu, Huangbiao and Ke, Xiao and Wu, Huanqi and Xu, Rui and Li, Yuezhou and Guo, Wenzhong},
    title     = {Language-Guided Audio-Visual Learning for Long-Term Sports Assessment},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {23967-23977}
}
```

## Acknowledgement
This repository builds upon [GDLT (CVPR 2022)](https://github.com/xuangch/CVPR22_GDLT).

We thank the authors for their contributions to the research community.