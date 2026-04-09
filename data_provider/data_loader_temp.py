    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h',
                 percent=100, model_name='gpt2', num_nodes=7):
        # size [seq_len, label_len, pred_len]
        self.percent = percent
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.num_nodes = num_nodes
        self.root_path = root_path
        self.data_path = data_path

        # 核心：解析嵌入文件路径，确保指向 ./Embeddings/600519_daily/train/ 这种干净路径
        data_name = os.path.basename(data_path).replace('.csv', '')
        self.embed_path = f"./Embeddings/{data_name}/{flag}/"

        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # 拼接数据读取路径
        read_path = self.data_path if self.data_path.endswith('.csv') else self.data_path + '.csv'
        df_raw = pd.read_csv(os.path.join(self.root_path, read_path))

        cols = list(df_raw.columns)
        if self.target in cols:
            cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # 核心修改：转换为收益率 (Returns)
        df_data = df_data.pct_change().fillna(0)

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
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
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len if hasattr(self, 'label_len') else s_end - 0
        r_end = r_begin + (self.label_len if hasattr(self, 'label_len') else 0) + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        embeddings_stack = []
        file_path = os.path.join(self.embed_path, f"{index}.h5")
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as hf:
                # 强制展平，并校验维度是否为 768 (LLM 特征)
                data = hf['embeddings'][:].flatten()
                if data.shape[0] != 768:
                    raise ValueError(f"CRITICAL ERROR: Found data size {data.shape[0]} at {file_path}, expected 768! Possible path mismatch.")
                
                tensor = torch.from_numpy(data).float()
                for _ in range(self.num_nodes):
                    embeddings_stack.append(tensor)
        else:
            raise FileNotFoundError(f"No embedding file found at {file_path}")
                
        embeddings = torch.stack(embeddings_stack, dim=0) 
        return seq_x, seq_y, seq_x_mark, seq_y_mark, embeddings
