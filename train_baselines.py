import argparse
import json
import os
import random
import time

import faulthandler
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_multistock import MultiStockSaveDataset
from models.baselines import ALSTMRegressor, LSTMRegressor, MLPRegressor, TransformerRegressor
from utils.metrics import MAE, metric, rank_ic, stock_prediction_stats


try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name", type=str, required=True, choices=["mlp", "lstm", "alstm", "transformer", "xgboost"])
    parser.add_argument("--data_path", type=str, default="dataset/HS300/all_stocks_complete_data.csv")
    parser.add_argument("--root_path", type=str, default=".")
    parser.add_argument("--freq", type=str, default="d")
    parser.add_argument("--task_name", type=str, default="multistock", choices=["multistock"])
    parser.add_argument("--stock_pool_file", type=str, default="")
    parser.add_argument("--start_date", type=str, default="")
    parser.add_argument("--end_date", type=str, default="")
    parser.add_argument("--train_start_date", type=str, default="")
    parser.add_argument("--train_end_date", type=str, default="")
    parser.add_argument("--val_start_date", type=str, default="")
    parser.add_argument("--val_end_date", type=str, default="")
    parser.add_argument("--trainval_start_date", type=str, default="")
    parser.add_argument("--trainval_end_date", type=str, default="")
    parser.add_argument("--test_start_date", type=str, default="")
    parser.add_argument("--test_end_date", type=str, default="")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--target_winsorize_lower", type=float, default=0.0)
    parser.add_argument("--target_winsorize_upper", type=float, default=1.0)
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--num_nodes", type=int, default=300)
    parser.add_argument("--input_features", type=int, default=9)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--target_horizon", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_n", type=float, default=0.2)
    parser.add_argument("--e_layer", type=int, default=3)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--es_patience", type=int, default=50)
    parser.add_argument("--xgb_estimators", type=int, default=400)
    parser.add_argument("--xgb_max_depth", type=int, default=6)
    parser.add_argument("--xgb_subsample", type=float, default=0.8)
    parser.add_argument("--xgb_colsample_bytree", type=float, default=0.8)
    parser.add_argument("--xgb_learning_rate", type=float, default=0.05)
    parser.add_argument("--xgb_early_stopping_rounds", type=int, default=30)
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    )
    return parser.parse_args()


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True


def load_data(args):
    if not args.stock_pool_file:
        raise ValueError("Multi-stock training requires --stock_pool_file.")

    dataset_kwargs = dict(
        root_path=args.root_path,
        stock_pool_file=args.stock_pool_file,
        size=[args.seq_len, 0, args.target_horizon],
        data_path=args.data_path,
        freq=args.freq,
        scale=True,
        target_horizon=args.target_horizon,
        start_date=args.start_date or None,
        end_date=args.end_date or None,
        train_start_date=args.train_start_date or None,
        train_end_date=args.train_end_date or None,
        val_start_date=args.val_start_date or None,
        val_end_date=args.val_end_date or None,
        trainval_start_date=args.trainval_start_date or None,
        trainval_end_date=args.trainval_end_date or None,
        test_start_date=args.test_start_date or None,
        test_end_date=args.test_end_date or None,
        val_ratio=args.val_ratio,
        target_winsorize_lower=args.target_winsorize_lower,
        target_winsorize_upper=args.target_winsorize_upper,
    )
    train_set = MultiStockSaveDataset(flag="train", **dataset_kwargs)
    val_set = MultiStockSaveDataset(flag="val", **dataset_kwargs)
    test_set = MultiStockSaveDataset(flag="test", **dataset_kwargs)

    args.num_nodes = train_set.num_nodes
    args.input_features = train_set.num_features

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def build_model(args):
    if args.model_name == "mlp":
        return MLPRegressor(
            seq_len=args.seq_len,
            input_features=args.input_features,
            hidden_dim=args.channel,
            dropout=args.dropout_n,
        )
    if args.model_name == "lstm":
        return LSTMRegressor(
            input_features=args.input_features,
            hidden_dim=args.channel,
            num_layers=args.e_layer,
            dropout=args.dropout_n,
        )
    if args.model_name == "alstm":
        return ALSTMRegressor(
            input_features=args.input_features,
            hidden_dim=args.channel,
            num_layers=args.e_layer,
            dropout=args.dropout_n,
        )
    if args.model_name == "transformer":
        return TransformerRegressor(
            seq_len=args.seq_len,
            input_features=args.input_features,
            hidden_dim=args.channel,
            num_layers=args.e_layer,
            num_heads=args.head,
            dropout=args.dropout_n,
        )
    raise ValueError(f"Unsupported deep baseline model: {args.model_name}")


class TorchBaselineTrainer:
    def __init__(self, model, learning_rate, weight_decay, device, epochs):
        self.model = model.to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=min(epochs, 50), eta_min=1e-6
        )
        self.loss = torch.nn.SmoothL1Loss()
        self.metric_mae = MAE
        self.clip = 5.0

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"The number of trainable parameters: {trainable_params}")
        print(f"The number of parameters: {total_params}")

    def train(self, input_tensor, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input_tensor)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.metric_mae(predict, real)
        return loss.item(), mae.item(), predict.detach()

    def eval(self, input_tensor, real):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input_tensor)
        loss = self.loss(predict, real)
        mae = self.metric_mae(predict, real)
        return loss.item(), mae.item(), predict.detach()


def _tensorize_batch(batch, device):
    x, y, _, _ = batch
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.float32, device=device),
    )


def run_epoch(engine, data_loader, device, training=False):
    losses = []
    maes = []
    rankics = []

    for batch in data_loader:
        x, y = _tensorize_batch(batch, device)
        if training:
            loss, mae, preds = engine.train(x, y)
        else:
            loss, mae, preds = engine.eval(x, y)
        losses.append(loss)
        maes.append(mae)
        rankics.append(rank_ic(preds, y).item())

    return float(np.mean(losses)), float(np.mean(maes)), float(np.mean(rankics))


def collect_predictions(engine, data_loader, device):
    predictions = []
    actuals = []
    for batch in data_loader:
        x, y = _tensorize_batch(batch, device)
        _, _, preds = engine.eval(x, y)
        predictions.append(preds.cpu())
        actuals.append(y.cpu())
    return torch.cat(predictions, dim=0), torch.cat(actuals, dim=0)


def summarize_predictions(preds, trues):
    mse, mae = metric(preds, trues)
    stats = stock_prediction_stats(preds, trues)
    stats.update({"mse": mse, "mae": mae})
    return stats


def result_path_from_args(args):
    data_name = os.path.basename(args.data_path).replace(".csv", "")
    result_path = os.path.join("./results", data_name)
    if args.run_tag:
        result_path = os.path.join(result_path, args.run_tag)
    return result_path


def save_outputs(args, test_set, preds, trues, metrics_payload):
    res_path = result_path_from_args(args)
    os.makedirs(res_path, exist_ok=True)
    np.save(os.path.join(res_path, "preds.npy"), preds)
    np.save(os.path.join(res_path, "trues.npy"), trues)
    np.save(os.path.join(res_path, "stock_codes.npy"), np.array(test_set.stock_codes, dtype=str))
    np.save(os.path.join(res_path, "target_dates.npy"), np.array(test_set.sample_target_dates, dtype=str))
    with open(os.path.join(res_path, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)
    print(f"测试结果已保存至: {res_path}")


def dataset_to_numpy(dataset):
    xs = []
    ys = []
    for index in range(len(dataset)):
        x, y, _, _ = dataset[index]
        xs.append(x.astype(np.float32))
        ys.append(y.astype(np.float32))
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def flatten_stock_samples(inputs, targets):
    num_samples, seq_len, num_nodes, num_features = inputs.shape
    flat_x = inputs.transpose(0, 2, 1, 3).reshape(num_samples * num_nodes, seq_len * num_features)
    flat_y = targets.reshape(num_samples * num_nodes)
    return flat_x, flat_y


def run_xgboost(args, train_set, val_set, test_set):
    if XGBRegressor is None:
        raise ImportError(
            "xgboost is not installed. Install it on the remote server before running the xgboost baseline."
        )

    train_x, train_y = dataset_to_numpy(train_set)
    val_x, val_y = dataset_to_numpy(val_set)
    test_x, test_y = dataset_to_numpy(test_set)

    flat_train_x, flat_train_y = flatten_stock_samples(train_x, train_y)
    flat_val_x, flat_val_y = flatten_stock_samples(val_x, val_y)
    flat_test_x, flat_test_y = flatten_stock_samples(test_x, test_y)

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        random_state=args.seed,
        n_jobs=args.num_workers,
        tree_method="hist",
    )

    fit_kwargs = {
        "eval_set": [(flat_val_x, flat_val_y)],
        "verbose": False,
    }
    if args.xgb_early_stopping_rounds > 0:
        fit_kwargs["early_stopping_rounds"] = args.xgb_early_stopping_rounds

    try:
        model.fit(flat_train_x, flat_train_y, **fit_kwargs)
    except TypeError:
        fit_kwargs.pop("early_stopping_rounds", None)
        model.fit(flat_train_x, flat_train_y, **fit_kwargs)

    val_pred = model.predict(flat_val_x).reshape(len(val_set), train_set.num_nodes)
    test_pred = model.predict(flat_test_x).reshape(len(test_set), train_set.num_nodes)

    val_pred_tensor = torch.from_numpy(val_pred).float()
    val_true_tensor = torch.from_numpy(val_y).float()
    test_pred_tensor = torch.from_numpy(test_pred).float()
    test_true_tensor = torch.from_numpy(test_y).float()

    val_mse, _ = metric(val_pred_tensor, val_true_tensor)
    summary = summarize_predictions(test_pred_tensor, test_true_tensor)

    best_iteration = getattr(model, "best_iteration", None)
    if best_iteration is None:
        best_iteration = getattr(model, "best_ntree_limit", 0)
    if not best_iteration:
        best_iteration = args.xgb_estimators

    metrics_payload = {
        "task_name": "multistock",
        "model_name": args.model_name,
        "data_path": args.data_path,
        "stock_pool_file": args.stock_pool_file,
        "graph_file": "",
        "graph_weight_transform": "none",
        "graph_top_k": 0,
        "graph_time_decay": "none",
        "graph_decay_half_life_days": 0.0,
        "target_winsorize_lower": args.target_winsorize_lower,
        "target_winsorize_upper": args.target_winsorize_upper,
        "embedding_tag": "",
        "run_tag": args.run_tag,
        "seq_len": args.seq_len,
        "target_horizon": args.target_horizon,
        "num_nodes": args.num_nodes,
        "input_features": args.input_features,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "best_epoch": int(best_iteration),
        "best_val_loss": float(val_mse),
        "test_mse": float(summary["mse"]),
        "test_mae": float(summary["mae"]),
        "test_ic": float(summary["ic"]),
        "test_ic_std": float(summary["ic_std"]),
        "test_icir": float(summary["icir"]),
        "test_rank_ic": float(summary["rank_ic"]),
        "test_rank_ic_std": float(summary["rank_ic_std"]),
        "test_rank_icir": float(summary["rank_icir"]),
    }

    print(
        f"Test MSE: {summary['mse']:.4f}, Test MAE: {summary['mae']:.4f}, "
        f"Test IC: {summary['ic']:.4f}, Test RankIC: {summary['rank_ic']:.4f}, "
        f"Test ICIR: {summary['icir']:.4f}, Test RankICIR: {summary['rank_icir']:.4f}"
    )
    save_outputs(args, test_set, test_pred, test_y, metrics_payload)


def run_torch_baseline(args, train_set, val_set, test_set, train_loader, val_loader, test_loader):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_name = os.path.basename(args.data_path).replace(".csv", "")

    save_root = os.path.join(args.save, data_name)
    if args.run_tag:
        save_root = os.path.join(save_root, args.run_tag)
    save_root = os.path.join(
        save_root,
        f"{args.model_name}_{args.seq_len}_{args.target_horizon}_{args.channel}_{args.e_layer}_"
        f"{args.learning_rate}_{args.dropout_n}_{args.seed}",
    )
    os.makedirs(save_root, exist_ok=True)
    best_model_path = os.path.join(save_root, "best_model.pth")

    model = build_model(args)
    engine = TorchBaselineTrainer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        epochs=args.epochs,
    )

    best_epoch = 0
    best_val_loss = float("inf")
    epochs_since_best = 0
    train_times = []
    val_times = []

    print(args)
    print(f"Resolved num_nodes={args.num_nodes}, input_features={args.input_features}")
    print(f"Start training baseline model: {args.model_name}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_start = time.time()
        train_loss, train_mae, train_rankic = run_epoch(engine, train_loader, device, training=True)
        train_times.append(time.time() - train_start)

        val_start = time.time()
        val_loss, val_mae, val_rankic = run_epoch(engine, val_loader, device, training=False)
        val_times.append(time.time() - val_start)

        print(f"Epoch: {epoch:03d}, Training Time: {train_times[-1]:.4f} secs")
        print(f"Epoch: {epoch:03d}, Validation Time: {val_times[-1]:.4f} secs")
        print(
            f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
            f"Train RankIC: {train_rankic:.4f}"
        )
        print(
            f"Epoch: {epoch:03d}, Valid Loss: {val_loss:.4f}, Valid MAE: {val_mae:.4f}, "
            f"Valid RankIC: {val_rankic:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(engine.model.state_dict(), best_model_path)
            print(f"Validation improved. Saved checkpoint to {best_model_path}")
        else:
            epochs_since_best += 1
            print("No update")

        engine.scheduler.step()

        if epochs_since_best >= args.es_patience and epoch >= max(args.epochs // 2, 1):
            break

    print(f"Average Training Time: {np.mean(train_times):.4f} secs/epoch")
    print(f"Average Validation Time: {np.mean(val_times):.4f} secs")
    print("Training ends")
    print(f"The epoch of the best result: {best_epoch}")
    print(f"The valid loss of the best model: {best_val_loss:.4f}")

    engine.model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_pre, test_real = collect_predictions(engine, test_loader, device)
    summary = summarize_predictions(test_pre, test_real)

    metrics_payload = {
        "task_name": "multistock",
        "model_name": args.model_name,
        "data_path": args.data_path,
        "stock_pool_file": args.stock_pool_file,
        "graph_file": "",
        "graph_weight_transform": "none",
        "graph_top_k": 0,
        "graph_time_decay": "none",
        "graph_decay_half_life_days": 0.0,
        "target_winsorize_lower": args.target_winsorize_lower,
        "target_winsorize_upper": args.target_winsorize_upper,
        "embedding_tag": "",
        "run_tag": args.run_tag,
        "seq_len": args.seq_len,
        "target_horizon": args.target_horizon,
        "num_nodes": args.num_nodes,
        "input_features": args.input_features,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "test_samples": len(test_set),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_mse": float(summary["mse"]),
        "test_mae": float(summary["mae"]),
        "test_ic": float(summary["ic"]),
        "test_ic_std": float(summary["ic_std"]),
        "test_icir": float(summary["icir"]),
        "test_rank_ic": float(summary["rank_ic"]),
        "test_rank_ic_std": float(summary["rank_ic_std"]),
        "test_rank_icir": float(summary["rank_icir"]),
    }
    print(
        f"Test MSE: {summary['mse']:.4f}, Test MAE: {summary['mae']:.4f}, "
        f"Test IC: {summary['ic']:.4f}, Test RankIC: {summary['rank_ic']:.4f}, "
        f"Test ICIR: {summary['icir']:.4f}, Test RankICIR: {summary['rank_icir']:.4f}"
    )
    save_outputs(args, test_set, test_pre.numpy(), test_real.numpy(), metrics_payload)


def main():
    args = parse_args()
    seed_it(args.seed)
    train_set, val_set, test_set, train_loader, val_loader, test_loader = load_data(args)

    if args.model_name == "xgboost":
        run_xgboost(args, train_set, val_set, test_set)
        return

    run_torch_baseline(args, train_set, val_set, test_set, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.4f}")
