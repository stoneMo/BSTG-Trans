"""
    metrics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.spatial.distance import pdist


def MAE(scores, targets):
    MAE = F.l1_loss(scores, targets)
    MAE = MAE.detach().item()
    return MAE

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def get_beta(batch_idx, m, beta_type, epoch, num_epochs):

    # 'Blundell', 'Standard', etc. Use float for const value

    if type(beta_type) is float:
        return beta_type
    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta

def compute_mue(pred, gt, uncertainty):
    B = gt.shape[0] 
    mue = 0
    for i in range(len(pred)):
        mue += MAE(pred[i], gt) * uncertainty[i].mean()
    return mue.detach().item()

def compute_diversity(pred, *args):
    pred = pred.detach().cpu().numpy()
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt, *args):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    pred = pred.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()