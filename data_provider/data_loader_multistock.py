import os
import warnings

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.graph_utils import load_stock_pool, normalize_stock_code, resolve_data_path
from utils.tools import StandardScaler

warnings.filterwarnings("ignore")


PRICE_COLUMNS = ["open", "high", "low", "close"]
VOLUME_COLUMNS = ["vol", "amount"]
DEFAULT_FEATURE_COLUMNS = PRICE_COLUMNS + VOLUME_COLUMNS + ["daily_return", "log_return", "amplitude"]


def _find_column(columns, candidates, required=True):
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    if required:
        raise ValueError(f"Missing required column. Tried candidates: {candidates}")
    return None


def _canonicalize_columns(df):
    rename_map = {
        _find_column(df.columns, ["date", "trade_date"]): "date",
        _find_column(df.columns, ["stock_code", "ts_code", "code", "ticker", "symbol"]): "stock_code",
        _find_column(df.columns, ["open"]): "open",
        _find_column(df.columns, ["high"]): "high",
        _find_column(df.columns, ["low"]): "low",
        _find_column(df.columns, ["close"]): "close",
        _find_column(df.columns, ["vol", "volume"]): "vol",
        _find_column(df.columns, ["amount", "turnover", "turnover_value"]): "amount",
    }
    return df.rename(columns=rename_map)


def _build_time_marks(date_index, timeenc=0):
    df_stamp = pd.DataFrame({"date": pd.to_datetime(date_index)})
    if timeenc == 0:
        df_stamp["year"] = df_stamp.date.dt.year
        df_stamp["month"] = df_stamp.date.dt.month
        df_stamp["day"] = df_stamp.date.dt.day
        df_stamp["weekday"] = df_stamp.date.dt.weekday
        df_stamp["hour"] = 0
        df_stamp["minute"] = 0
        return df_stamp.drop(columns=["date"]).to_numpy(dtype=np.float32)
    raise NotImplementedError("Only timeenc=0 is supported for the multi-stock loader.")


def _candidate_label_indices(total_length, seq_len, target_horizon):
    start = seq_len - 1
    stop = total_length - target_horizon
    if stop <= start:
        return np.empty(0, dtype=np.int64)
    return np.arange(start, stop, dtype=np.int64)


def _default_split_indices(total_length, seq_len, target_horizon):
    candidate_indices = _candidate_label_indices(total_length, seq_len, target_horizon)
    if len(candidate_indices) == 0:
        raise ValueError(
            f"Not enough dates to build samples for seq_len={seq_len} and target_horizon={target_horizon}."
        )

    num_train = int(total_length * 0.7)
    num_test = int(total_length * 0.2)
    num_val = total_length - num_train - num_test
    train_end = num_train - 1
    val_end = num_train + num_val - 1
    last_end = total_length - 1

    future_indices = candidate_indices + target_horizon
    train_indices = candidate_indices[future_indices <= train_end]
    val_indices = candidate_indices[(candidate_indices > train_end) & (future_indices <= val_end)]
    test_indices = candidate_indices[(candidate_indices > val_end) & (future_indices <= last_end)]

    split_map = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    return split_map, train_indices


def _date_based_split_indices(
    date_index,
    seq_len,
    target_horizon,
    flag,
    trainval_start_date=None,
    trainval_end_date=None,
    test_start_date=None,
    test_end_date=None,
    val_ratio=0.1,
):
    candidate_indices = _candidate_label_indices(len(date_index), seq_len, target_horizon)
    if len(candidate_indices) == 0:
        raise ValueError(
            f"Not enough dates to build samples for seq_len={seq_len} and target_horizon={target_horizon}."
        )

    dates = pd.DatetimeIndex(date_index)
    anchor_dates = dates[candidate_indices]
    future_dates = dates[candidate_indices + target_horizon]

    trainval_start = pd.to_datetime(trainval_start_date) if trainval_start_date else dates.min()
    trainval_end = pd.to_datetime(trainval_end_date) if trainval_end_date else dates.max()
    test_start = pd.to_datetime(test_start_date) if test_start_date else None
    test_end = pd.to_datetime(test_end_date) if test_end_date else dates.max()

    fit_mask = (anchor_dates >= trainval_start) & (anchor_dates <= trainval_end) & (future_dates <= trainval_end)
    if test_start is not None:
        fit_mask &= future_dates < test_start
    fit_indices = candidate_indices[fit_mask]
    if len(fit_indices) < 2:
        raise ValueError(
            "Not enough train/validation samples in the requested date window. "
            f"trainval_start_date={trainval_start}, trainval_end_date={trainval_end}"
        )

    if not 0 < float(val_ratio) < 1:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")

    val_count = max(1, int(round(len(fit_indices) * float(val_ratio))))
    if val_count >= len(fit_indices):
        val_count = len(fit_indices) - 1

    train_indices = fit_indices[:-val_count]
    val_indices = fit_indices[-val_count:]

    test_mask = np.ones(len(candidate_indices), dtype=bool)
    if test_start is not None:
        test_mask &= anchor_dates >= test_start
    if test_end is not None:
        test_mask &= anchor_dates <= test_end
        test_mask &= future_dates <= test_end
    if trainval_end_date:
        test_mask &= anchor_dates > trainval_end

    test_indices = candidate_indices[test_mask]

    split_map = {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }
    selected = split_map[flag]
    if len(selected) == 0:
        raise ValueError(
            f"No samples for split '{flag}' with trainval_end_date={trainval_end_date}, "
            f"test_start_date={test_start_date}, test_end_date={test_end_date}."
        )
    return split_map, train_indices


def _prepare_stock_frame(stock_df, all_dates):
    stock_df = stock_df.set_index("date").sort_index()
    stock_df = stock_df.reindex(all_dates)

    for column in PRICE_COLUMNS:
        stock_df[column] = pd.to_numeric(stock_df[column], errors="coerce").ffill().bfill()
    for column in VOLUME_COLUMNS:
        stock_df[column] = pd.to_numeric(stock_df[column], errors="coerce").fillna(0.0)

    if stock_df["close"].isna().any():
        raise ValueError("Close prices still contain NaNs after filling. Remove the affected stock from the pool.")

    prev_close = stock_df["close"].shift(1).replace(0, np.nan)
    stock_df["daily_return"] = stock_df["close"].pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    stock_df["log_return"] = np.log(stock_df["close"]).diff().replace([np.inf, -np.inf], 0.0).fillna(0.0)
    stock_df["amplitude"] = ((stock_df["high"] - stock_df["low"]) / prev_close).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return stock_df[DEFAULT_FEATURE_COLUMNS].astype(np.float32)


class _BaseMultiStockDataset(Dataset):
    def __init__(
        self,
        root_path,
        stock_pool_file,
        flag="train",
        size=None,
        data_path="multi_stock_panel.csv",
        scale=True,
        timeenc=0,
        freq="d",
        percent=100,
        target_horizon=5,
        start_date=None,
        end_date=None,
        trainval_start_date=None,
        trainval_end_date=None,
        test_start_date=None,
        test_end_date=None,
        val_ratio=0.1,
    ):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 60, 0, target_horizon
        else:
            self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]
        self.flag = flag
        self.root_path = root_path
        self.data_path = data_path
        self.stock_pool_file = stock_pool_file
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.target_horizon = target_horizon
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        self.trainval_start_date = trainval_start_date
        self.trainval_end_date = trainval_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date
        self.val_ratio = val_ratio
        self.scaler = StandardScaler()
        self.feature_names = list(DEFAULT_FEATURE_COLUMNS)
        self.prompt_feature_name = "close"
        self.prompt_feature_index = self.feature_names.index(self.prompt_feature_name)
        self.stock_codes = load_stock_pool(stock_pool_file)
        self.num_nodes = len(self.stock_codes)
        self.num_features = len(self.feature_names)
        data_name = os.path.basename(data_path).replace(".csv", "")
        self.embed_path = f"./Embeddings/{data_name}/{flag}/"
        self.__read_data__()

    def __read_data__(self):
        data_file = resolve_data_path(self.root_path, self.data_path)
        df_raw = pd.read_csv(data_file)
        df_raw = _canonicalize_columns(df_raw)
        df_raw["date"] = pd.to_datetime(df_raw["date"])
        df_raw["stock_code"] = df_raw["stock_code"].map(normalize_stock_code)
        if self.start_date is not None:
            df_raw = df_raw[df_raw["date"] >= self.start_date]
        if self.end_date is not None:
            df_raw = df_raw[df_raw["date"] <= self.end_date]
        df_raw = df_raw[df_raw["stock_code"].isin(self.stock_codes)].copy()

        if df_raw.empty:
            raise ValueError(
                "Panel data is empty after filtering. "
                f"Check date range start_date={self.start_date} end_date={self.end_date}."
            )

        missing_stocks = [code for code in self.stock_codes if code not in set(df_raw["stock_code"].unique())]
        if missing_stocks:
            raise ValueError(f"Panel data is missing stock codes from the pool: {missing_stocks}")

        all_dates = pd.Index(sorted(df_raw["date"].unique()))
        feature_panels = []
        close_panels = []
        for stock_code in self.stock_codes:
            stock_frame = _prepare_stock_frame(df_raw[df_raw["stock_code"] == stock_code], all_dates=all_dates)
            feature_panels.append(stock_frame.to_numpy(dtype=np.float32))
            close_panels.append(stock_frame["close"].to_numpy(dtype=np.float32))

        feature_panel = np.stack(feature_panels, axis=1)  # [T, N, F]
        close_panel = np.stack(close_panels, axis=1)  # [T, N]

        has_explicit_date_split = any(
            value for value in [self.trainval_start_date, self.trainval_end_date, self.test_start_date, self.test_end_date]
        )
        if has_explicit_date_split:
            split_map, train_indices = _date_based_split_indices(
                all_dates,
                seq_len=self.seq_len,
                target_horizon=self.target_horizon,
                flag=self.flag,
                trainval_start_date=self.trainval_start_date,
                trainval_end_date=self.trainval_end_date,
                test_start_date=self.test_start_date,
                test_end_date=self.test_end_date,
                val_ratio=self.val_ratio,
            )
        else:
            split_map, train_indices = _default_split_indices(
                total_length=len(all_dates),
                seq_len=self.seq_len,
                target_horizon=self.target_horizon,
            )

        if self.percent < 100 and self.flag == "train":
            keep = max(1, len(split_map["train"]) * self.percent // 100)
            split_map["train"] = split_map["train"][:keep]

        if len(split_map[self.flag]) == 0:
            raise ValueError(
                f"Split '{self.flag}' is empty after filtering for seq_len={self.seq_len} "
                f"and target_horizon={self.target_horizon}."
            )

        if self.scale:
            scaler_start = 0
            if self.trainval_start_date:
                trainval_start = pd.to_datetime(self.trainval_start_date)
                scaler_start = int(np.searchsorted(all_dates.to_numpy(), trainval_start.to_datetime64(), side="left"))
            scaler_end = train_indices[-1] + 1
            train_slice = feature_panel[scaler_start:scaler_end]
            self.scaler.fit(train_slice.reshape(-1, self.num_features))
            self.scaler.std = np.where(self.scaler.std == 0, 1.0, self.scaler.std)
            feature_panel = self.scaler.transform(feature_panel.reshape(-1, self.num_features)).reshape(feature_panel.shape)

        target_returns = np.zeros_like(close_panel, dtype=np.float32)
        max_base_index = len(close_panel) - self.target_horizon
        if max_base_index > 0:
            target_returns[:max_base_index] = (
                close_panel[self.target_horizon:] / np.clip(close_panel[:-self.target_horizon], a_min=1e-8, a_max=None)
            ) - 1.0

        self.data_x = feature_panel.astype(np.float32)
        self.data_y = target_returns.astype(np.float32)
        self.data_stamp = _build_time_marks(all_dates, timeenc=self.timeenc)
        self.sample_end_indices = split_map[self.flag].astype(np.int64)
        self.available_length = len(self.sample_end_indices)

    def __len__(self):
        return self.available_length

    def _get_target_mark(self, label_index):
        return self.data_stamp[label_index + self.target_horizon]


class MultiStockSaveDataset(_BaseMultiStockDataset):
    def __getitem__(self, index):
        label_index = int(self.sample_end_indices[index])
        s_begin, s_end = label_index - self.seq_len + 1, label_index + 1
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[label_index]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self._get_target_mark(label_index)
        return seq_x, seq_y, seq_x_mark, seq_y_mark


class MultiStockEmbeddingDataset(_BaseMultiStockDataset):
    def __getitem__(self, index):
        label_index = int(self.sample_end_indices[index])
        s_begin, s_end = label_index - self.seq_len + 1, label_index + 1
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[label_index]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self._get_target_mark(label_index)

        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing embedding file at {file_path}")

        with h5py.File(file_path, "r") as hf:
            data = hf["embeddings"][:]
            embeddings = torch.from_numpy(data).float()
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0).repeat(self.num_nodes, 1)
            elif embeddings.dim() == 2 and embeddings.shape[0] != self.num_nodes and embeddings.shape[1] == self.num_nodes:
                embeddings = embeddings.transpose(0, 1)
            elif embeddings.dim() != 2:
                raise ValueError(f"Unexpected embedding shape {tuple(embeddings.shape)} in {file_path}")

            if embeddings.shape[0] != self.num_nodes:
                raise ValueError(
                    f"Embedding node count mismatch in {file_path}: "
                    f"expected {self.num_nodes}, got {embeddings.shape[0]}"
                )

        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings
