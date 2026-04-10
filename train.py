import argparse
import os
import random
import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from data_provider.data_loader_emb import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute
from data_provider.data_loader_multistock import MultiStockEmbeddingDataset
from models.TimeCMA import Dual
from utils.graph_utils import (
    build_and_save_cooccurrence_graph,
    default_graph_output_path,
    infer_node_columns,
    load_graph_adjacency,
    load_stock_pool,
    resolve_data_path,
)
from utils.metrics import MAE, metric, rank_ic

import faulthandler

faulthandler.enable()
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:150"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda", help="")
    parser.add_argument("--data_path", type=str, default="ETTm1", help="data path")
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/", help="root path of the data file")
    parser.add_argument("--freq", type=str, default="h", help="frequency for time features encoding")
    parser.add_argument("--task_name", type=str, default="legacy", choices=["legacy", "multistock"])
    parser.add_argument("--stock_pool_file", type=str, default="", help="ordered stock pool file for multi-stock tasks")
    parser.add_argument("--start_date", type=str, default="", help="inclusive start date for multi-stock filtering")
    parser.add_argument("--end_date", type=str, default="", help="inclusive end date for multi-stock filtering")
    parser.add_argument("--trainval_start_date", type=str, default="", help="inclusive start date for train/val samples")
    parser.add_argument("--trainval_end_date", type=str, default="", help="inclusive end date for train/val samples")
    parser.add_argument("--test_start_date", type=str, default="", help="inclusive start date for test samples")
    parser.add_argument("--test_end_date", type=str, default="", help="inclusive end date for test samples")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="chronological validation ratio within train/val range")
    parser.add_argument("--channel", type=int, default=32, help="hidden size for stock/node representations")
    parser.add_argument("--num_nodes", type=int, default=7, help="number of nodes")
    parser.add_argument("--input_features", type=int, default=1, help="number of per-node input features")
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=96, help="legacy prediction length")
    parser.add_argument("--target_horizon", type=int, default=5, help="future return horizon for multi-stock tasks")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--dropout_n", type=float, default=0.2, help="dropout rate of neural network layers")
    parser.add_argument("--d_llm", type=int, default=768, help="hidden dimensions")
    parser.add_argument("--e_layer", type=int, default=1, help="layers of transformer encoder")
    parser.add_argument("--d_layer", type=int, default=1, help="layers of transformer decoder")
    parser.add_argument("--head", type=int, default=8, help="heads of attention")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="weight decay rate")
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--model_name", type=str, default="gpt2", help="llm")
    parser.add_argument("--epochs", type=int, default=100, help="")
    parser.add_argument("--graph_file", type=str, default="", help="path to a co-occurrence graph csv")
    parser.add_argument("--auto_build_graph", action="store_true", help="build the co-occurrence graph when it is missing")
    parser.add_argument("--rebuild_graph", action="store_true", help="force rebuilding the co-occurrence graph")
    parser.add_argument(
        "--graph_split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
        help="data split used to build the graph",
    )
    parser.add_argument("--graph_event_threshold", type=float, default=0.5, help="z-score threshold for graph event co-occurrence")
    parser.add_argument("--graph_min_weight", type=int, default=1, help="minimum edge weight kept in the graph")
    parser.add_argument("--graph_steps", type=int, default=1, help="number of graph propagation steps")
    parser.add_argument("--graph_alpha", type=float, default=0.5, help="initial residual mixing weight for graph propagation")
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument(
        "--es_patience",
        type=int,
        default=50,
        help="quit if no improvement after this many iterations",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="./logs/" + str(time.strftime("%Y-%m-%d-%H:%M:%S")) + "-",
        help="save path",
    )
    return parser.parse_args()


class trainer:
    def __init__(
        self,
        channel,
        num_nodes,
        seq_len,
        pred_len,
        dropout_n,
        d_llm,
        e_layer,
        d_layer,
        head,
        lrate,
        wdecay,
        device,
        epochs,
        graph_adjacency=None,
        graph_steps=1,
        graph_alpha=0.5,
        task_name="legacy",
        input_features=1,
    ):
        output_len = 1 if task_name == "multistock" else pred_len
        self.model = Dual(
            device=device,
            channel=channel,
            num_nodes=num_nodes,
            seq_len=seq_len,
            pred_len=output_len,
            dropout_n=dropout_n,
            d_llm=d_llm,
            e_layer=e_layer,
            d_layer=d_layer,
            head=head,
            graph_adjacency=graph_adjacency,
            graph_steps=graph_steps,
            graph_alpha=graph_alpha,
            task_name=task_name,
            input_features=input_features,
        )

        self.optimizer = optim.AdamW(self.model.parameters(), lr=lrate, weight_decay=wdecay)
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
    if args.task_name == "multistock":
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
            trainval_start_date=args.trainval_start_date or None,
            trainval_end_date=args.trainval_end_date or None,
            test_start_date=args.test_start_date or None,
            test_end_date=args.test_end_date or None,
            val_ratio=args.val_ratio,
        )
        train_set = MultiStockEmbeddingDataset(flag="train", **dataset_kwargs)
        val_set = MultiStockEmbeddingDataset(flag="val", **dataset_kwargs)
        test_set = MultiStockEmbeddingDataset(flag="test", **dataset_kwargs)
        args.num_nodes = train_set.num_nodes
        args.input_features = train_set.num_features
    else:
        data_map = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
        }
        data_class = data_map.get(args.data_path, Dataset_Custom)
        train_set = data_class(
            root_path=args.root_path,
            flag="train",
            scale=True,
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            freq=args.freq,
            num_nodes=args.num_nodes,
        )
        val_set = data_class(
            root_path=args.root_path,
            flag="val",
            scale=True,
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            freq=args.freq,
            num_nodes=args.num_nodes,
        )
        test_set = data_class(
            root_path=args.root_path,
            flag="test",
            scale=True,
            size=[args.seq_len, 0, args.pred_len],
            data_path=args.data_path,
            freq=args.freq,
            num_nodes=args.num_nodes,
        )
        args.input_features = 1

    scaler = getattr(train_set, "scaler", None)
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
    return train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler


def prepare_graph(args):
    if not args.graph_file and not args.auto_build_graph:
        return None, None

    graph_file = args.graph_file

    if args.task_name == "multistock":
        if not args.stock_pool_file:
            raise ValueError("Multi-stock graph loading requires --stock_pool_file.")
        if not graph_file:
            raise FileNotFoundError("Multi-stock mode requires an explicit --graph_file.")
        node_columns = load_stock_pool(args.stock_pool_file)
    else:
        if not graph_file:
            graph_file = default_graph_output_path(args.root_path, args.data_path, split=args.graph_split)
        need_build = args.auto_build_graph and (args.rebuild_graph or not os.path.exists(graph_file))
        if need_build:
            graph_file, _ = build_and_save_cooccurrence_graph(
                root_path=args.root_path,
                data_path=args.data_path,
                output_path=graph_file,
                split=args.graph_split,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                event_threshold=args.graph_event_threshold,
                min_weight=args.graph_min_weight,
            )
            print(f"Built co-occurrence graph: {graph_file}")
        data_file = resolve_data_path(args.root_path, args.data_path)
        node_columns = infer_node_columns(data_file)

    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")

    graph_adjacency = load_graph_adjacency(graph_file, node_columns=node_columns)
    return graph_adjacency, graph_file


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


def run_epoch(engine, data_loader, device, training=False, task_name="legacy"):
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
        if task_name == "multistock":
            rankics.append(rank_ic(preds, y).item())

    mean_rankic = float(np.mean(rankics)) if rankics else None
    return float(np.mean(losses)), float(np.mean(maes)), mean_rankic


def collect_predictions(engine, data_loader, device):
    predictions = []
    actuals = []
    for batch in data_loader:
        x, y, x_mark, _, embeddings = _tensorize_batch(batch, device)
        _, _, preds = engine.eval(x, x_mark, embeddings, y)
        predictions.append(preds.cpu())
        actuals.append(y.cpu())
    return torch.cat(predictions, dim=0), torch.cat(actuals, dim=0)


def summarize_predictions(preds, trues, task_name):
    if task_name == "multistock":
        mse, mae = metric(preds, trues)
        return {"mse": mse, "mae": mae, "rank_ic": rank_ic(preds, trues).item()}

    amse = []
    amae = []
    for horizon_index in range(preds.shape[1]):
        mse, mae = metric(preds[:, horizon_index], trues[:, horizon_index])
        amse.append(mse)
        amae.append(mae)
    return {"mse": float(np.mean(amse)), "mae": float(np.mean(amae))}


def result_path_from_args(args):
    data_name = os.path.basename(args.data_path).replace(".csv", "")
    return os.path.join("./results/", data_name)


def main():
    args = parse_args()
    seed_it(args.seed)

    train_set, val_set, test_set, train_loader, val_loader, test_loader, scaler = load_data(args)
    graph_adjacency, graph_file = prepare_graph(args)

    del scaler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = os.path.basename(args.data_path).replace(".csv", "")

    save_root = os.path.join(
        args.save,
        data_name,
        f"{args.task_name}_{args.seq_len}_{args.target_horizon if args.task_name == 'multistock' else args.pred_len}"
        f"_{args.channel}_{args.e_layer}_{args.d_layer}_{args.learning_rate}_{args.dropout_n}_{args.seed}",
    )
    os.makedirs(save_root, exist_ok=True)
    best_model_path = os.path.join(save_root, "best_model.pth")

    print(args)
    print(f"Resolved num_nodes={args.num_nodes}, input_features={args.input_features}")
    if graph_file:
        print(f"Using graph file: {graph_file}")

    engine = trainer(
        channel=args.channel,
        num_nodes=args.num_nodes,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        dropout_n=args.dropout_n,
        d_llm=args.d_llm,
        e_layer=args.e_layer,
        d_layer=args.d_layer,
        head=args.head,
        lrate=args.learning_rate,
        wdecay=args.weight_decay,
        device=device,
        epochs=args.epochs,
        graph_adjacency=graph_adjacency,
        graph_steps=args.graph_steps,
        graph_alpha=args.graph_alpha,
        task_name=args.task_name,
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
        train_loss, train_mae, train_rankic = run_epoch(
            engine, train_loader, device, training=True, task_name=args.task_name
        )
        train_times.append(time.time() - train_start)

        val_start = time.time()
        val_loss, val_mae, val_rankic = run_epoch(
            engine, val_loader, device, training=False, task_name=args.task_name
        )
        val_times.append(time.time() - val_start)

        print(f"Epoch: {epoch:03d}, Training Time: {train_times[-1]:.4f} secs")
        print(f"Epoch: {epoch:03d}, Validation Time: {val_times[-1]:.4f} secs")
        if args.task_name == "multistock":
            print(
                f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                f"Train RankIC: {train_rankic:.4f}"
            )
            print(
                f"Epoch: {epoch:03d}, Valid Loss: {val_loss:.4f}, Valid MAE: {val_mae:.4f}, "
                f"Valid RankIC: {val_rankic:.4f}"
            )
        else:
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            print(f"Epoch: {epoch:03d}, Valid Loss: {val_loss:.4f}, Valid MAE: {val_mae:.4f}")

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
    summary = summarize_predictions(test_pre, test_real, task_name=args.task_name)

    res_path = result_path_from_args(args)
    os.makedirs(res_path, exist_ok=True)
    np.save(os.path.join(res_path, "preds.npy"), test_pre.numpy())
    np.save(os.path.join(res_path, "trues.npy"), test_real.numpy())
    if args.task_name == "multistock":
        np.save(os.path.join(res_path, "stock_codes.npy"), np.array(test_set.stock_codes, dtype=object))
        print(
            f"Test MSE: {summary['mse']:.4f}, Test MAE: {summary['mae']:.4f}, "
            f"Test RankIC: {summary['rank_ic']:.4f}"
        )
    else:
        print(f"On average horizons, Test MSE: {summary['mse']:.4f}, Test MAE: {summary['mae']:.4f}")
    print(f"测试结果已保存至: {res_path}")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.4f}")
