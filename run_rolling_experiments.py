import argparse
import calendar
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import date


DEFAULT_METHODS = ("no_graph", "static", "dynamic6m")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market_data", default="dataset/HS300/all_stocks_complete_data.csv")
    parser.add_argument("--snowball_csv", default="dataset/Snowball/snowball_posts_clean_2011_2025.csv.gz")
    parser.add_argument("--stock_pool_file", default="dataset/HS300/stock_pool.csv")
    parser.add_argument("--runtime_root", default="dataset/HS300/rolling/runtime")
    parser.add_argument("--graph_root", default="dataset/HS300/rolling/graphs")
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS), choices=list(DEFAULT_METHODS))
    parser.add_argument("--test_month_start", default="2025-04")
    parser.add_argument("--test_month_end", default="2025-09")
    parser.add_argument("--train_months", type=int, default=12)
    parser.add_argument("--val_months", type=int, default=3)
    parser.add_argument("--dynamic_graph_months", type=int, default=6)
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--target_horizon", type=int, default=5)
    parser.add_argument("--freq", default="d")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--channel", type=int, default=128)
    parser.add_argument("--e_layer", type=int, default=3)
    parser.add_argument("--d_layer", type=int, default=2)
    parser.add_argument("--dropout_n", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--es_patience", type=int, default=30)
    parser.add_argument("--graph_steps", type=int, default=1)
    parser.add_argument("--graph_alpha", type=float, default=0.5)
    parser.add_argument("--graph_weight_transform", default="log1p", choices=["none", "log1p"])
    parser.add_argument("--graph_top_k", type=int, default=10)
    parser.add_argument("--dynamic_time_decay", default="exp", choices=["none", "exp"])
    parser.add_argument("--dynamic_decay_half_life_days", type=float, default=60.0)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_month(text):
    year_text, month_text = text.split("-", 1)
    return date(int(year_text), int(month_text), 1)


def add_months(value, offset):
    month_index = (value.year * 12 + value.month - 1) + offset
    year = month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def month_end(value):
    return date(value.year, value.month, calendar.monthrange(value.year, value.month)[1])


def iter_months(start_month, end_month):
    current = parse_month(start_month)
    end_value = parse_month(end_month)
    while current <= end_value:
        yield current
        current = add_months(current, 1)


def to_text(value):
    return value.isoformat()


def build_window(test_month, train_months, val_months):
    train_start = add_months(test_month, -(train_months + val_months))
    train_end = month_end(add_months(test_month, -(val_months + 1)))
    val_start = add_months(test_month, -val_months)
    val_end = month_end(add_months(test_month, -1))
    test_start = test_month
    test_end = month_end(test_month)
    window_id = f"test_{test_month.year}-{test_month.month:02d}"
    return {
        "window_id": window_id,
        "test_month": f"{test_month.year}-{test_month.month:02d}",
        "start_date": to_text(train_start),
        "end_date": to_text(test_end),
        "train_start_date": to_text(train_start),
        "train_end_date": to_text(train_end),
        "val_start_date": to_text(val_start),
        "val_end_date": to_text(val_end),
        "test_start_date": to_text(test_start),
        "test_end_date": to_text(test_end),
    }


def graph_dates_for_method(window, method, dynamic_graph_months):
    if method == "static":
        return window["train_start_date"], window["train_end_date"]
    if method == "dynamic6m":
        test_month = parse_month(window["test_month"])
        graph_start = add_months(test_month, -dynamic_graph_months)
        graph_end = month_end(add_months(test_month, -1))
        return to_text(graph_start), to_text(graph_end)
    raise ValueError(f"Method {method} does not use a graph window.")


def normalize_number_token(value):
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def graph_build_signature(args, method):
    if method == "static":
        return "decay-none"
    if method == "dynamic6m":
        if args.dynamic_time_decay == "none":
            return "decay-none"
        return f"decay-{args.dynamic_time_decay}-hl{normalize_number_token(args.dynamic_decay_half_life_days)}"
    return "no-graph"


def result_signature(args, method):
    if method == "no_graph":
        return "plain"

    parts = [
        f"gw-{args.graph_weight_transform}",
        f"topk-{args.graph_top_k}",
    ]
    if method == "dynamic6m":
        parts.append(graph_build_signature(args, method))
    return "_".join(parts)


def ensure_parent(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def run_command(cmd, cwd):
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path, payload):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def is_stale_metrics(metrics, args, method, runtime_graph):
    if method == "no_graph":
        return False
    graph_file = (metrics or {}).get("graph_file", "")
    if graph_file.replace("\\", "/") != runtime_graph.replace("\\", "/"):
        return True
    if (metrics or {}).get("graph_weight_transform", "none") != args.graph_weight_transform:
        return True
    if int((metrics or {}).get("graph_top_k", 0)) != int(args.graph_top_k):
        return True
    expected_decay = "none" if method == "static" else args.dynamic_time_decay
    if (metrics or {}).get("graph_time_decay", "none") != expected_decay:
        return True
    expected_half_life = 0.0 if method == "static" else float(args.dynamic_decay_half_life_days)
    if float((metrics or {}).get("graph_decay_half_life_days", 0.0)) != expected_half_life:
        return True
    return False


def maybe_build_graph(args, repo_root, window, method):
    if method == "no_graph":
        return ""

    graph_start, graph_end = graph_dates_for_method(window, method, args.dynamic_graph_months)
    graph_dir = os.path.join(args.graph_root, method, graph_build_signature(args, method))
    graph_path = os.path.join(graph_dir, f"{window['window_id']}.csv")
    if os.path.exists(os.path.join(repo_root, graph_path)) and not args.force:
        return graph_path

    ensure_parent(os.path.join(repo_root, graph_path))
    cmd = [
        args.python_bin,
        "build_snowball_cooccurrence_matrix.py",
        "--input_csv",
        args.snowball_csv,
        "--stock_pool_file",
        args.stock_pool_file,
        "--output_csv",
        graph_path,
        "--start_date",
        graph_start,
        "--end_date",
        graph_end,
    ]
    if method == "dynamic6m":
        cmd.extend(
            [
                "--time_decay",
                args.dynamic_time_decay,
                "--decay_half_life_days",
                str(args.dynamic_decay_half_life_days),
            ]
        )
    run_command(cmd, cwd=repo_root)
    return graph_path


def ensure_runtime_stock_pool(args, repo_root, window):
    runtime_dir = os.path.join(args.runtime_root, window["window_id"])
    stock_pool_runtime = os.path.join(runtime_dir, "stock_pool_runtime.csv")
    if os.path.exists(os.path.join(repo_root, stock_pool_runtime)) and not args.force:
        return stock_pool_runtime

    ensure_parent(os.path.join(repo_root, stock_pool_runtime))
    cmd = [
        args.python_bin,
        "build_runtime_assets.py",
        "--market_data",
        args.market_data,
        "--stock_pool_file",
        args.stock_pool_file,
        "--output_stock_pool",
        stock_pool_runtime,
    ]
    run_command(cmd, cwd=repo_root)
    return stock_pool_runtime


def ensure_runtime_graph(args, repo_root, window, method, graph_file):
    runtime_dir = os.path.join(args.runtime_root, window["window_id"])
    stock_pool_runtime = os.path.join(runtime_dir, "stock_pool_runtime.csv")
    runtime_graph_dir = os.path.join(runtime_dir, method, graph_build_signature(args, method))
    runtime_graph = os.path.join(runtime_graph_dir, "graph_runtime.csv")
    if os.path.exists(os.path.join(repo_root, runtime_graph)) and not args.force:
        return stock_pool_runtime, runtime_graph

    ensure_parent(os.path.join(repo_root, runtime_graph))
    cmd = [
        args.python_bin,
        "build_runtime_assets.py",
        "--market_data",
        args.market_data,
        "--stock_pool_file",
        args.stock_pool_file,
        "--graph_file",
        graph_file,
        "--output_stock_pool",
        stock_pool_runtime,
        "--output_graph",
        runtime_graph,
    ]
    run_command(cmd, cwd=repo_root)
    return stock_pool_runtime, runtime_graph


def embedding_marker_path(repo_root, data_path, embedding_tag):
    data_name = os.path.basename(data_path).replace(".csv", "")
    return os.path.join(repo_root, "Embeddings", data_name, embedding_tag, "_complete.json")


def ensure_embeddings(args, repo_root, window, stock_pool_runtime):
    embedding_tag = os.path.join("rolling_experiments", window["window_id"])
    marker_path = embedding_marker_path(repo_root, args.market_data, embedding_tag)
    if os.path.exists(marker_path) and not args.force:
        return embedding_tag

    for divide in ("train", "val", "test"):
        cmd = [
            args.python_bin,
            "storage/store_emb.py",
            "--task_name",
            "multistock",
            "--divide",
            divide,
            "--root_path",
            ".",
            "--data_path",
            args.market_data,
            "--stock_pool_file",
            stock_pool_runtime,
            "--freq",
            args.freq,
            "--input_len",
            str(args.seq_len),
            "--output_len",
            str(args.target_horizon),
            "--target_horizon",
            str(args.target_horizon),
            "--start_date",
            window["start_date"],
            "--end_date",
            window["end_date"],
            "--train_start_date",
            window["train_start_date"],
            "--train_end_date",
            window["train_end_date"],
            "--val_start_date",
            window["val_start_date"],
            "--val_end_date",
            window["val_end_date"],
            "--test_start_date",
            window["test_start_date"],
            "--test_end_date",
            window["test_end_date"],
            "--embedding_tag",
            embedding_tag,
            "--batch_size",
            "1",
            "--num_workers",
            str(args.num_workers),
        ]
        run_command(cmd, cwd=repo_root)

    marker_payload = {
        "window_id": window["window_id"],
        "embedding_tag": embedding_tag,
        "market_data": args.market_data,
        "stock_pool_file": stock_pool_runtime,
    }
    dump_json(marker_path, marker_payload)
    return embedding_tag


def run_training(args, repo_root, window, method, stock_pool_runtime, embedding_tag, runtime_graph):
    run_tag = os.path.join("rolling_experiments", method, result_signature(args, method), window["window_id"])
    data_name = os.path.basename(args.market_data).replace(".csv", "")
    metrics_path = os.path.join(repo_root, "results", data_name, run_tag, "metrics.json")
    if os.path.exists(metrics_path) and not args.force:
        existing_metrics = load_json(metrics_path)
        if not is_stale_metrics(existing_metrics, args, method, runtime_graph):
            return run_tag, existing_metrics

    cmd = [
        args.python_bin,
        "train.py",
        "--task_name",
        "multistock",
        "--data_path",
        args.market_data,
        "--root_path",
        ".",
        "--stock_pool_file",
        stock_pool_runtime,
        "--seq_len",
        str(args.seq_len),
        "--target_horizon",
        str(args.target_horizon),
        "--start_date",
        window["start_date"],
        "--end_date",
        window["end_date"],
        "--train_start_date",
        window["train_start_date"],
        "--train_end_date",
        window["train_end_date"],
        "--val_start_date",
        window["val_start_date"],
        "--val_end_date",
        window["val_end_date"],
        "--test_start_date",
        window["test_start_date"],
        "--test_end_date",
        window["test_end_date"],
        "--batch_size",
        str(args.batch_size),
        "--freq",
        args.freq,
        "--learning_rate",
        str(args.learning_rate),
        "--channel",
        str(args.channel),
        "--e_layer",
        str(args.e_layer),
        "--d_layer",
        str(args.d_layer),
        "--dropout_n",
        str(args.dropout_n),
        "--weight_decay",
        str(args.weight_decay),
        "--epochs",
        str(args.epochs),
        "--es_patience",
        str(args.es_patience),
        "--graph_steps",
        str(args.graph_steps),
        "--graph_alpha",
        str(args.graph_alpha),
        "--graph_weight_transform",
        args.graph_weight_transform,
        "--graph_top_k",
        str(args.graph_top_k),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--embedding_tag",
        embedding_tag,
        "--run_tag",
        run_tag,
    ]
    if runtime_graph:
        cmd.extend(["--graph_file", runtime_graph])
        if method == "dynamic6m":
            cmd.extend(
                [
                    "--graph_time_decay",
                    args.dynamic_time_decay,
                    "--graph_decay_half_life_days",
                    str(args.dynamic_decay_half_life_days),
                ]
            )
        else:
            cmd.extend(["--graph_time_decay", "none", "--graph_decay_half_life_days", "0"])
    run_command(cmd, cwd=repo_root)
    return run_tag, load_json(metrics_path)


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return statistics.pstdev(values)


def write_csv(path, fieldnames, rows):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_results(repo_root, market_data, rows):
    data_name = os.path.basename(market_data).replace(".csv", "")
    summary_root = os.path.join(repo_root, "results", data_name, "rolling_experiments")
    summary_rows = []
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        mse_values = [row["test_mse"] for row in method_rows]
        mae_values = [row["test_mae"] for row in method_rows]
        ic_values = [row["test_ic"] for row in method_rows if row["test_ic"] == row["test_ic"]]
        icir_values = [row["test_icir"] for row in method_rows if row["test_icir"] == row["test_icir"]]
        rankic_values = [row["test_rank_ic"] for row in method_rows if row["test_rank_ic"] == row["test_rank_ic"]]
        rankicir_values = [row["test_rank_icir"] for row in method_rows if row["test_rank_icir"] == row["test_rank_icir"]]
        summary_rows.append(
            {
                "method": method,
                "num_windows": len(method_rows),
                "avg_mse": sum(mse_values) / len(mse_values),
                "std_mse": safe_std(mse_values),
                "avg_mae": sum(mae_values) / len(mae_values),
                "std_mae": safe_std(mae_values),
                "avg_ic": sum(ic_values) / len(ic_values) if ic_values else math.nan,
                "std_ic": safe_std(ic_values) if ic_values else math.nan,
                "avg_icir": sum(icir_values) / len(icir_values) if icir_values else math.nan,
                "std_icir": safe_std(icir_values) if icir_values else math.nan,
                "avg_rank_ic": sum(rankic_values) / len(rankic_values) if rankic_values else math.nan,
                "std_rank_ic": safe_std(rankic_values) if rankic_values else math.nan,
                "avg_rank_icir": sum(rankicir_values) / len(rankicir_values) if rankicir_values else math.nan,
                "std_rank_icir": safe_std(rankicir_values) if rankicir_values else math.nan,
            }
        )

    detail_fields = [
        "method",
        "window_id",
        "test_month",
        "train_start_date",
        "train_end_date",
        "val_start_date",
        "val_end_date",
        "test_start_date",
        "test_end_date",
        "graph_start_date",
        "graph_end_date",
        "graph_weight_transform",
        "graph_top_k",
        "graph_time_decay",
        "graph_decay_half_life_days",
        "graph_source_file",
        "graph_file",
        "stock_pool_file",
        "embedding_tag",
        "run_tag",
        "train_samples",
        "val_samples",
        "test_samples",
        "best_epoch",
        "best_val_loss",
        "test_mse",
        "test_mae",
        "test_ic",
        "test_ic_std",
        "test_icir",
        "test_rank_ic",
        "test_rank_ic_std",
        "test_rank_icir",
    ]
    summary_fields = [
        "method",
        "num_windows",
        "avg_mse",
        "std_mse",
        "avg_mae",
        "std_mae",
        "avg_ic",
        "std_ic",
        "avg_icir",
        "std_icir",
        "avg_rank_ic",
        "std_rank_ic",
        "avg_rank_icir",
        "std_rank_icir",
    ]
    write_csv(os.path.join(summary_root, "window_metrics.csv"), detail_fields, rows)
    write_csv(os.path.join(summary_root, "method_summary.csv"), summary_fields, summary_rows)
    dump_json(
        os.path.join(summary_root, "window_metrics.json"),
        {
            "windows": rows,
            "summary": summary_rows,
        },
    )


def main():
    args = parse_args()
    repo_root = os.path.abspath(os.path.dirname(__file__))
    windows = [build_window(month, args.train_months, args.val_months) for month in iter_months(args.test_month_start, args.test_month_end)]

    rows = []
    for window in windows:
        print(f"=== Window {window['window_id']} ===", flush=True)
        stock_pool_runtime = ensure_runtime_stock_pool(args, repo_root, window)
        embedding_tag = ensure_embeddings(args, repo_root, window, stock_pool_runtime)

        for method in args.methods:
            runtime_graph = ""
            graph_start = ""
            graph_end = ""
            graph_file = ""
            if method != "no_graph":
                graph_file = maybe_build_graph(args, repo_root, window, method)
                graph_start, graph_end = graph_dates_for_method(window, method, args.dynamic_graph_months)
                stock_pool_runtime, runtime_graph = ensure_runtime_graph(args, repo_root, window, method, graph_file)

            run_tag, metrics = run_training(
                args=args,
                repo_root=repo_root,
                window=window,
                method=method,
                stock_pool_runtime=stock_pool_runtime,
                embedding_tag=embedding_tag,
                runtime_graph=runtime_graph,
            )
            rows.append(
                {
                    "method": method,
                    "window_id": window["window_id"],
                    "test_month": window["test_month"],
                    "train_start_date": window["train_start_date"],
                    "train_end_date": window["train_end_date"],
                    "val_start_date": window["val_start_date"],
                    "val_end_date": window["val_end_date"],
                    "test_start_date": window["test_start_date"],
                    "test_end_date": window["test_end_date"],
                    "graph_start_date": graph_start,
                    "graph_end_date": graph_end,
                    "graph_weight_transform": metrics.get("graph_weight_transform", args.graph_weight_transform),
                    "graph_top_k": metrics.get("graph_top_k", args.graph_top_k),
                    "graph_time_decay": metrics.get(
                        "graph_time_decay",
                        "none" if method == "static" else (args.dynamic_time_decay if method == "dynamic6m" else "none"),
                    ),
                    "graph_decay_half_life_days": metrics.get(
                        "graph_decay_half_life_days",
                        0.0 if method != "dynamic6m" else args.dynamic_decay_half_life_days,
                    ),
                    "graph_source_file": graph_file,
                    "graph_file": metrics.get("graph_file", graph_file),
                    "stock_pool_file": metrics.get("stock_pool_file", stock_pool_runtime),
                    "embedding_tag": metrics.get("embedding_tag", embedding_tag),
                    "run_tag": metrics.get("run_tag", run_tag),
                    "train_samples": metrics.get("train_samples", 0),
                    "val_samples": metrics.get("val_samples", 0),
                    "test_samples": metrics.get("test_samples", 0),
                    "best_epoch": metrics.get("best_epoch", 0),
                    "best_val_loss": metrics.get("best_val_loss", math.nan),
                    "test_mse": metrics.get("test_mse", math.nan),
                    "test_mae": metrics.get("test_mae", math.nan),
                    "test_ic": metrics.get("test_ic", math.nan),
                    "test_ic_std": metrics.get("test_ic_std", math.nan),
                    "test_icir": metrics.get("test_icir", math.nan),
                    "test_rank_ic": metrics.get("test_rank_ic", math.nan),
                    "test_rank_ic_std": metrics.get("test_rank_ic_std", math.nan),
                    "test_rank_icir": metrics.get("test_rank_icir", math.nan),
                }
            )

    summarize_results(repo_root, args.market_data, rows)
    print("Rolling experiments completed.", flush=True)


if __name__ == "__main__":
    main()
