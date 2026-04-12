import os
import re

import numpy as np
import pandas as pd
import torch


ETT_HOUR_DATASETS = {"ETTh1", "ETTh2"}
ETT_MINUTE_DATASETS = {"ETTm1", "ETTm2"}
STOCK_CODE_PATTERN = re.compile(r"(\d{6})")


def resolve_data_path(root_path, data_path):
    file_name = data_path if data_path.endswith(".csv") else f"{data_path}.csv"
    return os.path.abspath(os.path.join(root_path, file_name))


def normalize_stock_code(value):
    if value is None:
        return ""
    text = str(value).strip()
    match = STOCK_CODE_PATTERN.search(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text.zfill(6)
    return text


def load_stock_pool(stock_pool_file):
    stock_pool_path = os.path.abspath(stock_pool_file)
    if not os.path.exists(stock_pool_path):
        raise FileNotFoundError(f"Stock pool file not found: {stock_pool_path}")

    if stock_pool_path.endswith(".csv"):
        df = pd.read_csv(stock_pool_path, dtype=str)
        if "stock_code" in df.columns:
            codes = df["stock_code"].tolist()
        else:
            codes = df.iloc[:, 0].tolist()
    else:
        with open(stock_pool_path, "r", encoding="utf-8") as handle:
            codes = [line.strip().split(",")[0] for line in handle if line.strip()]

    ordered_codes = []
    seen = set()
    for code in codes:
        code = normalize_stock_code(code)
        if not code:
            continue
        if code not in seen:
            ordered_codes.append(code)
            seen.add(code)

    if not ordered_codes:
        raise ValueError(f"Stock pool file is empty: {stock_pool_path}")
    return ordered_codes


def infer_node_columns(data_file, time_col="date"):
    df = pd.read_csv(data_file, nrows=1)
    return [column for column in df.columns if column != time_col]


def default_graph_output_path(root_path, data_path, split="train"):
    data_file = resolve_data_path(root_path, data_path)
    stem, _ = os.path.splitext(data_file)
    return f"{stem}_cooccurrence_{split}.csv"


def _slice_dataframe(df, data_name, split, seq_len, pred_len):
    if split == "all":
        return df.copy()

    total = len(df)
    if data_name in ETT_HOUR_DATASETS:
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif data_name in ETT_MINUTE_DATASETS:
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(total * 0.7)
        num_test = int(total * 0.2)
        num_val = total - num_train - num_test
        border1s = [0, num_train - seq_len, total - num_test - seq_len]
        border2s = [num_train, num_train + num_val, total]

    split_to_index = {"train": 0, "val": 1, "test": 2}
    start = max(0, border1s[split_to_index[split]])
    end = min(total, border2s[split_to_index[split]])
    return df.iloc[start:end].copy()


def _series_event_signal(series, column_name, target="OT"):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    name = column_name.lower()

    if column_name == target:
        signal = values.pct_change()
    elif "vol" in name or "amount" in name:
        signal = np.log1p(values.clip(lower=0)).diff()
    elif "pct" in name or "change" in name:
        signal = values
    elif any(token in name for token in ("rsi", "kdj", "turnover", "macd", "boll")):
        signal = values.diff()
    else:
        signal = values.pct_change()

    return signal.replace([np.inf, -np.inf], 0).fillna(0.0)


def build_cooccurrence_matrix(
    df,
    node_columns,
    target="OT",
    event_threshold=0.5,
    min_weight=1,
):
    signal_frame = pd.DataFrame(
        {column: _series_event_signal(df[column], column_name=column, target=target) for column in node_columns}
    ).fillna(0.0)

    std = signal_frame.std(axis=0).replace(0, np.nan)
    normalized = signal_frame.divide(std, axis=1).replace([np.inf, -np.inf], 0).fillna(0.0)
    states = np.where(normalized.values >= event_threshold, 1, np.where(normalized.values <= -event_threshold, -1, 0))

    up_mask = (states == 1).astype(np.int32)
    down_mask = (states == -1).astype(np.int32)
    cooccurrence = up_mask.T @ up_mask + down_mask.T @ down_mask

    matrix = pd.DataFrame(cooccurrence, index=node_columns, columns=node_columns, dtype=np.int32)
    np.fill_diagonal(matrix.values, 0)
    if min_weight > 1:
        matrix = matrix.where(matrix >= min_weight, 0)
    return matrix


def build_and_save_cooccurrence_graph(
    root_path,
    data_path,
    output_path=None,
    split="train",
    seq_len=96,
    pred_len=96,
    target="OT",
    event_threshold=0.5,
    min_weight=1,
):
    data_file = resolve_data_path(root_path, data_path)
    df = pd.read_csv(data_file)
    data_name = os.path.splitext(os.path.basename(data_file))[0]
    node_columns = [column for column in df.columns if column != "date"]
    sliced_df = _slice_dataframe(df, data_name=data_name, split=split, seq_len=seq_len, pred_len=pred_len)
    matrix = build_cooccurrence_matrix(
        sliced_df,
        node_columns=node_columns,
        target=target,
        event_threshold=event_threshold,
        min_weight=min_weight,
    )

    save_path = os.path.abspath(output_path or default_graph_output_path(root_path, data_path, split=split))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    matrix.to_csv(save_path)
    return save_path, node_columns


def load_graph_adjacency(graph_file, node_columns, add_self_loops=True):
    graph_df = pd.read_csv(graph_file, index_col=0)
    graph_df.index = [normalize_stock_code(value) for value in graph_df.index]
    graph_df.columns = [normalize_stock_code(value) for value in graph_df.columns]
    node_columns = [normalize_stock_code(column) for column in node_columns]
    missing_columns = [column for column in node_columns if column not in graph_df.columns or column not in graph_df.index]
    if missing_columns:
        raise ValueError(f"Graph file {graph_file} is missing node columns: {missing_columns}")

    graph_df = graph_df.loc[node_columns, node_columns]
    matrix = graph_df.to_numpy(dtype=np.float32)
    matrix = np.maximum(matrix, 0.0)
    matrix = 0.5 * (matrix + matrix.T)

    if add_self_loops:
        matrix = matrix + np.eye(len(node_columns), dtype=np.float32)

    degree = matrix.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float32)
    valid_degree = degree > 0
    inv_sqrt_degree[valid_degree] = np.power(degree[valid_degree], -0.5)
    normalized = inv_sqrt_degree[:, None] * matrix * inv_sqrt_degree[None, :]
    return torch.tensor(normalized, dtype=torch.float32)
