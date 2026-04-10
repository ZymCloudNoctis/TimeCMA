import numpy as np
import torch

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return torch.mean(torch.abs(pred - true))

def MSE(pred, true):
    return torch.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    mse = MSE(pred, true).item()
    mae = MAE(pred, true).item()
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    
    # return mae,mse,rmse,mape,mspe
    return mse,mae


def _rank_tensor(values):
    order = torch.argsort(values, dim=-1)
    ranks = torch.argsort(order, dim=-1).float()
    return ranks


def rank_ic(pred, true):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        true = true.unsqueeze(0)

    pred_ranks = _rank_tensor(pred)
    true_ranks = _rank_tensor(true)

    pred_centered = pred_ranks - pred_ranks.mean(dim=-1, keepdim=True)
    true_centered = true_ranks - true_ranks.mean(dim=-1, keepdim=True)

    numerator = (pred_centered * true_centered).sum(dim=-1)
    denominator = torch.sqrt(
        torch.clamp((pred_centered ** 2).sum(dim=-1) * (true_centered ** 2).sum(dim=-1), min=1e-12)
    )
    correlations = numerator / denominator
    return correlations.mean()
