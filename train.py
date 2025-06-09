import numpy as np
from scipy.stats import spearmanr
from utils import AverageMeter


def train_epoch(epoch, model, loss_fn, train_loader, optim, logger, device, args):
    model.train()
    preds = np.array([])
    labels = np.array([])
    losses = AverageMeter('loss', logger)

    for i, (video_feat, audio_feat, label) in enumerate(train_loader):
        video_feat = video_feat.to(device)      # (b, t, c)
        audio_feat = audio_feat.to(device)  # (b, t, c)
        label = label.float().to(device)
        out = model(video_feat,audio_feat)
        pred = out['output']
        loss = loss_fn(pred, label, out['embed'], out['embed2'], out['embed3'], out['embed4'], out['embed5'], out['embed6'], out['logits1'], out['logits2'], args)
        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss, label.shape[0])

        if len(preds) == 0:
            preds = pred.cpu().detach().numpy()
            labels = label.cpu().detach().numpy()
        else:
            preds = np.concatenate((preds, pred.cpu().detach().numpy()), axis=0)
            labels = np.concatenate((labels, label.cpu().detach().numpy()), axis=0)

    coef, _ = spearmanr(preds, labels)
    if logger is not None:
        logger.add_scalar('train coef', coef, epoch)
    avg_loss = losses.done(epoch)
    return avg_loss, coef
