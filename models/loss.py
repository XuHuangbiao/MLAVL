import torch
from torch import nn
from models.triplet_loss import HardTripletLoss
import torch.nn.functional as F
import numpy as np
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        batch_size, num_features, feat_dim = features.shape
        features = features.view(batch_size * num_features, feat_dim)
        labels = torch.arange(batch_size * num_features).cuda()
        similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)
        similarity_matrix = similarity_matrix / self.temperature
        loss = F.cross_entropy(similarity_matrix, labels)
        return loss


# 各种分布特征聚合
class LossFun(nn.Module):
    def __init__(self, alpha, margin):
        super(LossFun, self).__init__()
        self.mse_loss = nn.MSELoss()
        # self.mse_loss = nn.L1Loss()
        self.triplet_loss = HardTripletLoss(margin=margin, hardest=True)
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda()
        # self.infonce_loss = InfoNCELoss()
        self.alpha = alpha
        self.alpha_ce = 0.01

    def forward(self, pred, label, feat, feat2, feat3, feat4, feat5, feat6, clip, clip2, args):
        # feat (b, n, c), x (b, t, c)
        if feat is not None:
            device = feat.device
            b, n, c = feat.shape
            flat_feat = feat.reshape(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            t_loss = self.triplet_loss(flat_feat, la)
            # info_loss = self.infonce_loss(feat)
            # t_loss = t_loss + info_loss
            # t_loss = pair_diversity_loss(feat)
            if feat2 is not None:
                device = feat2.device
                b, n, c = feat2.shape
                flat_feat2 = feat2.reshape(-1, c)  # (bn, c)
                la2 = torch.arange(n, device=device).repeat(b)

                t_loss += self.triplet_loss(flat_feat2, la2)
            if feat3 is not None:
                device = feat3.device
                b, n, c = feat3.shape
                flat_feat = feat3.reshape(-1, c)  # (bn, c)
                la = torch.arange(n, device=device).repeat(b)

                t_loss += self.triplet_loss(flat_feat, la)
            if feat4 is not None:
                device = feat4.device
                b, n, c = feat4.shape
                flat_feat = feat4.reshape(-1, c)  # (bn, c)
                la = torch.arange(n, device=device).repeat(b)

                t_loss += self.triplet_loss(flat_feat, la)
            if feat5 is not None:
                device = feat5.device
                b, n, c = feat5.shape
                flat_feat = feat5.reshape(-1, c)  # (bn, c)
                la = torch.arange(n, device=device).repeat(b)

                t_loss += self.triplet_loss(flat_feat, la)
            if feat6 is not None:
                device = feat6.device
                b, n, c = feat6.shape
                flat_feat = feat6.reshape(-1, c)  # (bn, c)
                la = torch.arange(n, device=device).repeat(b)

                t_loss += self.triplet_loss(flat_feat, la)
        else:
            self.alpha = 0
            t_loss = 0
        if clip is not None:
            device = clip.device
            b, n, c = clip.shape
            flat_feat = clip.view(-1, c)  # (bn, c)
            la = torch.arange(n, device=device).repeat(b)

            ce_loss = self.ce_loss(flat_feat, la)
            if clip2 is not None:
                device = clip2.device
                b, n, c = clip2.shape
                flat_feat2 = clip2.view(-1, c)  # (bn, c)
                la2 = torch.arange(n, device=device).repeat(b)

                ce_loss += self.ce_loss(flat_feat2, la2)
        #     if clip3 is not None:
        #         device = clip3.device
        #         b, n, c = clip3.shape
        #         flat_feat3 = clip3.view(-1, c)  # (bn, c)
        #         la3 = torch.arange(n, device=device).repeat(b)
        #
        #         ce_loss += self.ce_loss(flat_feat3, la3)
        else:
            self.alpha_ce = 0
            ce_loss = 0
        mse_loss = 10. * self.mse_loss(pred, label)
        # mse_loss = 10. * (lam * self.mse_loss(pred, label) + (1 - lam) * self.mse_loss(pred, label2))
        # mse_loss = pearson_loss(pred, label)
        return mse_loss + self.alpha * t_loss + self.alpha_ce * ce_loss
        # return mse_loss + self.alpha * t_loss + self.alpha_ce * ce_loss, mse_loss, t_loss, ce_loss
        # return f_loss, mse_loss, t_loss, f_loss
