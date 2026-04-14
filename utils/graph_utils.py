import os
import re

import numpy as np
import pandas as pd
import torch


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
        if not code or code in seen:
            continue
        ordered_codes.append(code)
        seen.add(code)

    if not ordered_codes:
        raise ValueError(f"Stock pool file is empty: {stock_pool_path}")
    return ordered_codes


def _apply_top_k_filter(matrix, top_k):
    top_k = int(top_k)
    if top_k <= 0 or matrix.shape[0] <= 1:
        return matrix

    filtered = np.zeros_like(matrix, dtype=np.float32)
    for row_idx in range(matrix.shape[0]):
        row = matrix[row_idx].copy()
        row[row_idx] = 0.0
        positive_indices = np.flatnonzero(row > 0)
        if len(positive_indices) <= top_k:
            filtered[row_idx, positive_indices] = row[positive_indices]
            continue

        candidate_values = row[positive_indices]
        top_positions = np.argpartition(candidate_values, -top_k)[-top_k:]
        keep_indices = positive_indices[top_positions]
        filtered[row_idx, keep_indices] = row[keep_indices]

    filtered = np.maximum(filtered, filtered.T)
    np.fill_diagonal(filtered, 0.0)
    return filtered


def load_graph_adjacency(graph_file, node_columns, add_self_loops=True, weight_transform="none", top_k=0):
    graph_df = pd.read_csv(graph_file, index_col=0)
    graph_df.index = [normalize_stock_code(value) for value in graph_df.index]
    graph_df.columns = [normalize_stock_code(value) for value in graph_df.columns]
    node_columns = [normalize_stock_code(column) for column in node_columns]

    missing_columns = [column for column in node_columns if column not in graph_df.columns or column not in graph_df.index]
    if missing_columns:
        raise ValueError(f"Graph file {graph_file} is missing node columns: {missing_columns}")

    matrix = graph_df.loc[node_columns, node_columns].to_numpy(dtype=np.float32)
    matrix = np.maximum(matrix, 0.0)
    matrix = 0.5 * (matrix + matrix.T)
    np.fill_diagonal(matrix, 0.0)

    if weight_transform == "log1p":
        matrix = np.log1p(matrix)
    elif weight_transform != "none":
        raise ValueError(f"Unsupported graph weight transform: {weight_transform}")

    matrix = _apply_top_k_filter(matrix, top_k=top_k)

    if add_self_loops:
        matrix = matrix + np.eye(len(node_columns), dtype=np.float32)

    degree = matrix.sum(axis=1)
    inv_sqrt_degree = np.zeros_like(degree, dtype=np.float32)
    valid_degree = degree > 0
    inv_sqrt_degree[valid_degree] = np.power(degree[valid_degree], -0.5)
    normalized = inv_sqrt_degree[:, None] * matrix * inv_sqrt_degree[None, :]
    return torch.tensor(normalized, dtype=torch.float32)
