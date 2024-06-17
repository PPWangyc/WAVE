import torch
import numpy as np
from torchvision.models import ViT_H_14_Weights, vit_h_14
from torchmetrics.functional import accuracy
from skimage.metrics import structural_similarity as ssim

def batch_corrcoef(x, y):
    """
    Compute the correlation coefficient for each pair of rows in x and y.
    x and y should both have shape [N, K].
    Returns a tensor of shape [N] containing correlation coefficients.
    """
    # Ensure the tensors have the same shape
    assert x.shape == y.shape
    x, y = x.flatten(1), y.flatten(1)

    # compute the pearson correlation coefficient
    vx = x - torch.mean(x, dim=1, keepdim=True)
    vy = y - torch.mean(y, dim=1, keepdim=True)
    cost = torch.sum(vx * vy, dim=1) / (torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)))
    return cost

def masked_mse_loss(predict, target, mask):
    """
    Compute masked mean squared error between prediction and target.
    
    Parameters:
    - predict (torch.Tensor): Predicted tensor of shape [N, L]
    - target (torch.Tensor): Ground truth tensor of shape [N, L]
    - mask (torch.Tensor): Binary mask indicating regions to compute the loss, of shape [N, L]

    Returns:
    - torch.Tensor: Scalar tensor representing the loss
    """
    
    # Ensure the tensors have the same shape
    assert predict.shape == target.shape

    # Compute squared error
    loss = (predict - target) ** 2

    # Mean squared error per patch
    loss = loss.mean(dim=-1)

    # Compute mean loss only on masked regions
    final_loss = (loss * mask).sum() / mask.sum()

    return final_loss

def masked_l1_loss(predict, target, mask):
    """
    Compute masked mean absolute error between prediction and target.
    
    Parameters:
    - predict (torch.Tensor): Predicted tensor of shape [N, L]
    - target (torch.Tensor): Ground truth tensor of shape [N, L]
    - mask (torch.Tensor): Binary mask indicating regions to compute the loss, of shape [N, L]

    Returns:
    - torch.Tensor: Scalar tensor representing the loss
    """
    
    # Ensure the tensors have the same shape
    assert predict.shape == target.shape

    # Compute squared error
    loss = torch.abs(predict - target)

    # Mean squared error per patch
    loss = loss.mean(dim=-1)

    # Compute mean loss only on masked regions
    final_loss = (loss * mask).sum() / mask.sum()

    return final_loss

def l2_regularization(loss, model):
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    loss = loss + l2_reg * 1e-5
    return loss

def getRMSSD(rawData, sample_rate, segment_width=30, segment_overlap=0):
    _, measures = hp.process_segmentwise(rawData, sample_rate, segment_width=segment_width, 
                                         calc_freq=True, segment_overlap=segment_overlap)
    # lnHF_HRV = np.log(measures['hf'])
    rmssd=np.log(measures['rmssd'])
    return rmssd

def cleanRMSSD(rmssd):
    # remove nan in rmssd
    original_len = len(rmssd)
    rmssd = rmssd[~np.isnan(rmssd)]
    # interpolate rmssd to original length
    rmssd = np.interp(np.linspace(0, 1, original_len), np.linspace(0, 1, len(rmssd)), rmssd)
    return rmssd

def batchwise_cosine_similarity(Z,B):
    # https://www.h4pz.co/blog/2021/4/2/batch-cosine-similarity-in-pytorch-or-numpy-jax-cupy-etc
    B = B.T
    Z_norm = torch.linalg.norm(Z, dim=1, keepdim=True)  # Size (n, 1).
    B_norm = torch.linalg.norm(B, dim=0, keepdim=True)  # Size (1, b).
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity

def topk(similarities,labels,k=5):
    if k > similarities.shape[0]:
        k = similarities.shape[0]
    topsum=0
    for i in range(k):
        topsum += torch.sum(torch.argsort(similarities,axis=1)[:,-(i+1)] == labels)/len(labels)
    return topsum

def soft_clip_loss(preds, targs, temp=0.125):
    clip_clip = (targs @ targs.T)/temp
    brain_clip = (preds @ targs.T)/temp
    
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    
    loss = (loss1 + loss2)/2
    return loss

@torch.no_grad()
def n_way_top_k_acc(pred, class_id, n_way, num_trials=40, top_k=1):
    pick_range =[i for i in np.arange(len(pred)) if i != class_id]
    acc_list = []
    for t in range(num_trials):
        idxs_picked = np.random.choice(pick_range, n_way-1, replace=False)
        pred_picked = torch.cat([pred[class_id].unsqueeze(0), pred[idxs_picked]])
        acc = accuracy(pred_picked.unsqueeze(0), torch.tensor([0], device=pred.device), 
                    task='multiclass',num_classes=n_way,top_k=top_k)
        acc_list.append(acc.item())
    return np.mean(acc_list), np.std(acc_list)

def ssim_metric(img1, img2):
    return ssim(img1, img2, data_range=255, channel_axis=-1)

def pcc_metric(img1, img2):
    return np.corrcoef(img1.reshape(-1), img2.reshape(-1))[0, 1]

def mse_metric(img1, img2):
    return (np.square(img1 - img2)).mean()