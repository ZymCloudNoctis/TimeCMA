import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import h5py
import warnings

warnings.filterwarnings('ignore')

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path="./dataset/ETT-small/", flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, model_name='gpt2', num_nodes=7):
        if size == None:
            self.seq_len, self.label_len, self.pred_len = 24 * 4 * 4, 24 * 4, 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features, self.target, self.scale, self.timeenc, self.freq = features, target, scale, timeenc, freq
        self.root_path, self.data_path, self.num_nodes = root_path, data_path, num_nodes
        data_name = data_path.replace('.csv', '')
        self.embed_path = f"./Embeddings/{data_name}/{flag}/"
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]
        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0)
        self.data_x, self.data_y, self.data_stamp = data[border1:border2], data[border1:border2], data_stamp

    def __getitem__(self, index):
        s_begin, s_end = index, index + self.seq_len
        r_begin, r_end = s_end - self.label_len, s_end + self.pred_len
        seq_x, seq_y = self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]
        seq_x_mark, seq_y_mark = self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end]
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if not os.path.exists(file_path):
            print(f"Error: {os.path.abspath(file_path)} NOT FOUND")
            print(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Missing embedding file at {file_path}")
            
        with h5py.File(file_path, 'r') as hf:
            data = hf['embeddings'][:]
            embeddings = torch.from_numpy(data).float().squeeze(0)
            if embeddings.dim() == 1: embeddings = embeddings.unsqueeze(-1).repeat(1, self.num_nodes)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path="./dataset/ETT-small/", flag='train', size=None,
                 features='M', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t',
                 percent=100, model_name='gpt2', num_nodes=7):
        if size == None:
            self.seq_len, self.label_len, self.pred_len = 24 * 4 * 4, 24 * 4, 24 * 4
        else:
            self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features, self.target, self.scale, self.timeenc, self.freq = features, target, scale, timeenc, freq
        self.root_path, self.data_path, self.num_nodes = root_path, data_path, num_nodes
        data_name = data_path.replace('.csv', '')
        self.embed_path = f"./Embeddings/{data_name}/{flag}/"
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]
        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0)
        self.data_x, self.data_y, self.data_stamp = data[border1:border2], data[border1:border2], data_stamp

    def __getitem__(self, index):
        s_begin, s_end = index, index + self.seq_len
        r_begin, r_end = s_end - self.label_len, s_end + self.pred_len
        seq_x, seq_y = self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]
        seq_x_mark, seq_y_mark = self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end]
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if not os.path.exists(file_path):
            print(f"Error: {os.path.abspath(file_path)} NOT FOUND")
            print(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Missing embedding file at {file_path}")
            
        with h5py.File(file_path, 'r') as hf:
            data = hf['embeddings'][:]
            embeddings = torch.from_numpy(data).float().squeeze(0)
            if embeddings.dim() == 1: embeddings = embeddings.unsqueeze(-1).repeat(1, self.num_nodes)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, model_name='gpt2', num_nodes=7):
        self.percent = percent
        if size == None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 96
        else:
            self.seq_len, self.label_len, self.pred_len = size[0], size[1], size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features, self.target, self.scale, self.timeenc, self.freq, self.num_nodes = features, target, scale, timeenc, freq, num_nodes
        self.root_path, self.data_path = root_path, data_path
        data_name = os.path.basename(data_path).replace('.csv', '')
        self.embed_path = f"./Embeddings/{data_name}/{flag}/"
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        read_path = self.data_path if self.data_path.endswith('.csv') else self.data_path + '.csv'
        df_raw = pd.read_csv(os.path.join(self.root_path, read_path))
        num_train, num_test = int(len(df_raw) * 0.7), int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len
        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]
        
        # 核心修复：针对不同类型的列采用不同的预处理策略
        for col in df_data.columns:
            c = col.lower()
            if col == self.target:
                # 目标列(价格)保持原始水平，不转收益率，直接进行后续标准化
                # 这样预测结果就能直接匹配价格走势图
                pass
            elif 'pct' in c or 'change' in c or 'rsi' in c or 'kdj' in c or 'turnover' in c:
                # 已经是变动值或有界指标，保持原样
                pass
            elif 'vol' in c or 'amount' in c:
                # 成交量/成交额列，使用 log 缩放
                df_data[col] = np.log1p(df_data[col])
            else:
                # 输入的价格特征转为收益率，增强特征平稳性
                df_data[col] = df_data[col].pct_change()
        
        df_data = df_data.replace([np.inf, -np.inf], 0).fillna(0)

        if self.scale:
            self.scaler.fit(df_data[border1s[0]:border2s[0]].values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['year'] = df_stamp.date.apply(lambda row: row.year)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq).transpose(1, 0)
        self.data_x, self.data_y, self.data_stamp = data[border1:border2], data[border1:border2], data_stamp

    def __getitem__(self, index):
        s_begin, s_end = index, index + self.seq_len
        r_begin, r_end = s_end - self.label_len, s_end + self.pred_len
        seq_x, seq_y = self.data_x[s_begin:s_end], self.data_y[r_begin:r_end]
        seq_x_mark, seq_y_mark = self.data_stamp[s_begin:s_end], self.data_stamp[r_begin:r_end]
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if not os.path.exists(file_path):
            print(f"Error: {os.path.abspath(file_path)} NOT FOUND")
            print(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Missing embedding file at {file_path}")
            
        with h5py.File(file_path, 'r') as hf:
            raw_data = hf['embeddings'][:].flatten()
            
            # 自动化维度探测与修复逻辑
            if raw_data.shape[0] == 768:
                # 标准情况：正好是 768 维
                tensor = torch.from_numpy(raw_data).float()
            elif raw_data.shape[0] % 768 == 0:
                # 兼容情况：文件里存了多个节点的合并数据 (如 5376)，只取第一组 768
                tensor = torch.from_numpy(raw_data[:768]).float()
            else:
                # 无法自动修复的情况
                raise ValueError(f"CRITICAL DIMENSION MISMATCH: Found {raw_data.shape[0]} at {file_path}. Expected 768 or a multiple. Please re-run Store_600519_daily.sh")
            
            # 统一分发给所有节点
            embeddings = torch.stack([tensor] * self.num_nodes, dim=-1) # [768, Nodes]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
