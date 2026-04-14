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

from data_provider.data_loader_multistock import MultiStockEmbeddingDataset
from models.TimeCMA import Dual
from utils.graph_utils import load_graph_adjacency, load_stock_pool
from utils.metrics import MAE, metric, rank_ic, stock_prediction_stats


faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
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
    parser.add_argument("--embedding_tag", type=str, default="")
    parser.add_argument("--run_tag", type=str, default="")
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--num_nodes", type=int, default=300)
    parser.add_argument("--input_features", type=int, default=9)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--target_horizon", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--dropout_n", type=float, default=0.2)
    parser.add_argument("--d_llm", type=int, default=768)
    parser.add_argument("--e_layer", type=int, default=3)
    parser.add_argument("--d_layer", type=int, default=2)
    parser.add_argument("--head", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--graph_file", type=str, default="")
    parser.add_argument("--graph_steps", type=int, default=1)
    parser.add_argument("--graph_alpha", type=float, default=0.5)
    parser.add_argument("--graph_weight_transform", type=str, default="none", choices=["none", "log1p"])
    parser.add_argument("--graph_top_k", type=int, default=0)
    parser.add_argument("--graph_time_decay", type=str, default="none", choices=["none", "exp"])
    parser.add_argument("--graph_decay_half_life_days", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--es_patience", type=int, default=50)
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
    )
    return parser.parse_args()


class Trainer:
    def __init__(
        self,
        channel,
        num_nodes,
        seq_len,
        dropout_n,
        d_llm,
        e_layer,
        d_layer,
        head,
        learning_rate,
        weight_decay,
        device,
        epochs,
        graph_adjacency=None,
        graph_steps=1,
        graph_alpha=0.5,
        input_features=1,
    ):
        self.model = Dual(
            device=device,
            channel=channel,
            num_nodes=num_nodes,
            seq_len=seq_len,
            pred_len=1,
            dropout_n=dropout_n,
            d_llm=d_llm,
            e_layer=e_layer,
            d_layer=d_layer,
            head=head,
            graph_adjacency=graph_adjacency,
            graph_steps=graph_steps,
            graph_alpha=graph_alpha,
            task_name="multistock",
            input_features=input_features,
        )
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=min(epochs, 50), eta_min=1e-6
        )
        self.loss = torch.nn.SmoothL1Loss()
        self.metric_mae = MAE
        self.clip = 5
        print(f"The number of trainable parameters: {self.model.count_trainable_params()}")
        print(f"The number of parameters: {self.model.param_num()}")

    def train(self, input_tensor, mark_tensor, embeddings, real):
        self.model.train()
        self.optimizer.zero_grad()
        predict = self.model(input_tensor, mark_tensor, embeddings)
        loss = self.loss(predict, real)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae = self.metric_mae(predict, real)
        return loss.item(), mae.item(), predict.detach()

    def eval(self, input_tensor, mark_tensor, embeddings, real):
        self.model.eval()
        with torch.no_grad():
            predict = self.model(input_tensor, mark_tensor, embeddings)
        loss = self.loss(predict, real)
        mae = self.metric_mae(predict, real)
        return loss.item(), mae.item(), predict.detach()


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
        embedding_tag=args.embedding_tag or "",
    )
    train_set = MultiStockEmbeddingDataset(flag="train", **dataset_kwargs)
    val_set = MultiStockEmbeddingDataset(flag="val", **dataset_kwargs)
    test_set = MultiStockEmbeddingDataset(flag="test", **dataset_kwargs)

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


def prepare_graph(args):
    if not args.graph_file:
        return None, ""
    if not args.stock_pool_file:
        raise ValueError("Graph loading requires --stock_pool_file.")
    if not os.path.exists(args.graph_file):
        raise FileNotFoundError(f"Graph file not found: {args.graph_file}")

    node_columns = load_stock_pool(args.stock_pool_file)
    graph_adjacency = load_graph_adjacency(
        args.graph_file,
        node_columns=node_columns,
        weight_transform=args.graph_weight_transform,
        top_k=args.graph_top_k,
    )
    return graph_adjacency, args.graph_file


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


def _tensorize_batch(batch, device):
    x, y, x_mark, y_mark, embeddings = batch
    return (
        torch.as_tensor(x, dtype=torch.float32, device=device),
        torch.as_tensor(y, dtype=torch.float32, device=device),
        torch.as_tensor(x_mark, dtype=torch.float32, device=device),
        torch.as_tensor(y_mark, dtype=torch.float32, device=device),
        torch.as_tensor(embeddings, dtype=torch.float32, device=device),
    )


def run_epoch(engine, data_loader, device, training=False):
    losses = []
    maes = []
    rankics = []

    for batch in data_loader:
        x, y, x_mark, _, embeddings = _tensorize_batch(batch, device)
        if training:
            loss, mae, preds = engine.train(x, x_mark, embeddings, y)
        else:
            loss, mae, preds = engine.eval(x, x_mark, embeddings, y)
        losses.append(loss)
        maes.append(mae)
        rankics.append(rank_ic(preds, y).item())

    return float(np.mean(losses)), float(np.mean(maes)), float(np.mean(rankics))


def collect_predictions(engine, data_loader, device):
    predictions = []
    actuals = []
    for batch in data_loader:
        x, y, x_mark, _, embeddings = _tensorize_batch(batch, device)
        _, _, preds = engine.eval(x, x_mark, embeddings, y)
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


def main():
    args = parse_args()
    seed_it(args.seed)

    train_set, val_set, test_set, train_loader, val_loader, test_loader = load_data(args)
    graph_adjacency, graph_file = prepare_graph(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data_name = os.path.basename(args.data_path).replace(".csv", "")

    save_root = os.path.join(args.save, data_name)
    if args.run_tag:
        save_root = os.path.join(save_root, args.run_tag)
    save_root = os.path.join(
        save_root,
        f"multistock_{args.seq_len}_{args.target_horizon}_{args.channel}_{args.e_layer}_"
        f"{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}",
    )
    os.makedirs(save_root, exist_ok=True)
    best_model_path = os.path.join(save_root, "best_model.pth")

    print(args)
    print(f"Resolved num_nodes={args.num_nodes}, input_features={args.input_features}")
    if graph_file:
        print(f"Using graph file: {graph_file}")
    else:
        print("Running without a graph.")

    engine = Trainer(
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        epochs=args.epochs,
        graph_adjacency=graph_adjacency,
        graph_steps=args.graph_steps,
        graph_alpha=args.graph_alpha,
        input_features=args.input_features,
    )

    best_epoch = 0
    best_val_loss = float("inf")
    epochs_since_best = 0
    train_times = []
    val_times = []

    print("Start training...", flush=True)

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

    res_path = result_path_from_args(args)
    os.makedirs(res_path, exist_ok=True)
    np.save(os.path.join(res_path, "preds.npy"), test_pre.numpy())
    np.save(os.path.join(res_path, "trues.npy"), test_real.numpy())
    np.save(os.path.join(res_path, "stock_codes.npy"), np.array(test_set.stock_codes, dtype=str))
    np.save(os.path.join(res_path, "target_dates.npy"), np.array(test_set.sample_target_dates, dtype=str))

    metrics_payload = {
        "task_name": "multistock",
        "data_path": args.data_path,
        "stock_pool_file": args.stock_pool_file,
        "graph_file": graph_file or "",
        "graph_weight_transform": args.graph_weight_transform,
        "graph_top_k": args.graph_top_k,
        "graph_time_decay": args.graph_time_decay,
        "graph_decay_half_life_days": args.graph_decay_half_life_days,
        "target_winsorize_lower": args.target_winsorize_lower,
        "target_winsorize_upper": args.target_winsorize_upper,
        "embedding_tag": args.embedding_tag,
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
    with open(os.path.join(res_path, "metrics.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)
    print(f"测试结果已保存至: {res_path}")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.4f}")
