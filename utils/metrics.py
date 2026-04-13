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


def _center_last_dim(values):
    return values - values.mean(dim=-1, keepdim=True)


def _safe_corr(centered_pred, centered_true):
    numerator = (centered_pred * centered_true).sum(dim=-1)
    denominator = torch.sqrt(
        torch.clamp((centered_pred ** 2).sum(dim=-1) * (centered_true ** 2).sum(dim=-1), min=1e-12)
    )
    return numerator / denominator


def _mean_std_ir(values):
    if values.numel() == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "ir": nan}

    mean = values.mean()
    std = values.std(unbiased=False)
    if torch.isclose(std, torch.tensor(0.0, device=values.device, dtype=values.dtype)):
        ir = torch.tensor(float("nan"), device=values.device, dtype=values.dtype)
    else:
        ir = mean / std
    return {"mean": mean.item(), "std": std.item(), "ir": ir.item()}


def _rank_tensor(values):
    order = torch.argsort(values, dim=-1)
    ranks = torch.argsort(order, dim=-1).float()
    return ranks


def daily_ic_series(pred, true):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        true = true.unsqueeze(0)
    pred_centered = _center_last_dim(pred.float())
    true_centered = _center_last_dim(true.float())
    return _safe_corr(pred_centered, true_centered)


def rank_ic(pred, true):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        true = true.unsqueeze(0)

    pred_ranks = _rank_tensor(pred)
    true_ranks = _rank_tensor(true)

    pred_centered = _center_last_dim(pred_ranks)
    true_centered = _center_last_dim(true_ranks)

    correlations = _safe_corr(pred_centered, true_centered)
    return correlations.mean()


def daily_rank_ic_series(pred, true):
    if pred.dim() == 1:
        pred = pred.unsqueeze(0)
        true = true.unsqueeze(0)

    pred_ranks = _rank_tensor(pred)
    true_ranks = _rank_tensor(true)

    pred_centered = _center_last_dim(pred_ranks)
    true_centered = _center_last_dim(true_ranks)
    return _safe_corr(pred_centered, true_centered)


def stock_prediction_stats(pred, true):
    daily_ic = daily_ic_series(pred, true)
    daily_rank_ic = daily_rank_ic_series(pred, true)
    ic_stats = _mean_std_ir(daily_ic)
    rank_ic_stats = _mean_std_ir(daily_rank_ic)
    return {
        "ic": ic_stats["mean"],
        "ic_std": ic_stats["std"],
        "icir": ic_stats["ir"],
        "rank_ic": rank_ic_stats["mean"],
        "rank_ic_std": rank_ic_stats["std"],
        "rank_icir": rank_ic_stats["ir"],
    }
