from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn.functional as F
import pandas as pd


class FS1000Dataset(Dataset):
    def __init__(self, video_feat_path, audio_feat_path, label_path, action_list, clip_num=26, action_type='Ball', train=True, args=None):
        self.train = train
        self.video_path = video_feat_path
        self.audio_path = audio_feat_path
        self.erase_path = video_feat_path + '_erTrue'
        score_idx = {'TES': 130, 'PCS': 60, 'SS': 10, 'TR': 10, 'PE': 10, 'CO': 10, 'IN': 10}
        self.score_range = score_idx[action_type]
        args.score_range = self.score_range
        self.clip_num = clip_num
        self.labels = self.read_label(label_path, action_type)
        classes_all = pd.read_csv(action_list)
        self.action_list = classes_all['name'].values.tolist()

    @property
    def classes(self):
        return self.action_list

    def read_label(self, label_path, action_type):
        fr = open(label_path, 'r')
        idx = {'TES': 1, 'PCS': 2, 'SS': 3, 'TR': 4, 'PE': 5, 'CO': 6, 'IN': 7}
        labels = []
        score = []
        for i, line in enumerate(fr):
            line = line.strip().split()
            s = float(line[idx[action_type]])
            if action_type == "PCS":
                s = s / float(line[8])
            labels.append([line[0], s])
            score.append(s)
        print("max:", max(score))
        return labels

    def __getitem__(self, idx):
        video_feat = np.load(os.path.join(self.video_path, self.labels[idx][0] + '.npy'))
        audio_feat = np.load(os.path.join(self.audio_path, self.labels[idx][0] + '.npy'))

        # temporal random crop or padding
        video_feat = video_feat.mean(1)
        if self.train:
            if len(video_feat) > self.clip_num:
                st = np.random.randint(0, len(video_feat) - self.clip_num)
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
        else:
            if len(video_feat) > self.clip_num:
                st = (len(video_feat) - self.clip_num) // 2
                video_feat = video_feat[st:st + self.clip_num]
                audio_feat = audio_feat[st:st + self.clip_num]
                # erase_feat = erase_feat[st:st + self.clip_num]
            elif len(video_feat) < self.clip_num:
                new_feat = np.zeros((self.clip_num, video_feat.shape[1]))
                new_feat[:video_feat.shape[0]] = video_feat
                video_feat = new_feat
                new_feat = np.zeros((self.clip_num, audio_feat.shape[1]))
                new_feat[:audio_feat.shape[0]] = audio_feat
                audio_feat = new_feat
        audio_feat = torch.from_numpy(audio_feat).float()
        video_feat = torch.from_numpy(video_feat).float()
        return video_feat, audio_feat, self.normalize_score(self.labels[idx][1])
        # return video_feat, audio_feat, self.labels[idx][1]

    def __len__(self):
        return len(self.labels)

    def normalize_score(self, score):
        return score / self.score_range
