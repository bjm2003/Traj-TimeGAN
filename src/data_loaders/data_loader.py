import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class BaseDataLoader:
    def __init__(self):
        self.batch_size = 128
        self.shuffle = True
        self.num_workers = 4

    def _create_scaler(self, normalization_type):
        if normalization_type == 'zscore':
            return StandardScaler()
        elif normalization_type == 'minmax':
            return MinMaxScaler()
        raise ValueError(f"Unsupported normalization: {normalization_type}")


class SineDataLoader(BaseDataLoader):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        self.sample_length = 100
        self.noise_level = 0.05
        
        raw_data = np.load(self.data_path)['data']
        self.data = self._add_noise(raw_data)

    def _add_noise(self, data):
        noise = np.random.normal(scale=self.noise_level, size=data.shape)
        return data + noise

    def get_dataset(self):
        return self.data

    def get_loader(self):
        return DataLoader(self.data, 
                         batch_size=self.batch_size,
                         shuffle=self.shuffle,
                         num_workers=self.num_workers)

class HGVDataLoader(BaseDataLoader):
    def __init__(self, data_path, dataset_type='hgv'):
        super().__init__()
        self.data_path = data_path
        self.dataset_type = dataset_type
        
        # 根据数据集类型设置参数
        if self.dataset_type == 'hgv':
            self.normalization = 'minmax'
            self.channels = [0,1,2,3,4,5]
            self.channel_configs = [
                {'channels': [0,1,2], 'normalization': 'zscore'},
                {'channels': [3,4,5], 'normalization': 'minmax'}
            ]

        else:
            self.normalization = 'standard'
            self.channels = [0]
            self.channel_configs = [{'channels': [0], 'normalization': 'standard'}]

        
        raw_data = np.load(self.data_path)['data']
        print(f"原始数据形状：{raw_data.shape}")
        
        self.data = self._preprocess(raw_data)

    def _preprocess(self, data):
        # 按通道配置处理数据
        processed_segments = []
        for config in self.channel_configs:
            segment = data[..., config['channels']]
            scaler = self._create_scaler(config['normalization'])
            processed_segments.append(scaler.fit_transform(segment.reshape(-1, len(config['channels']))).reshape(segment.shape))
        
        # 合并处理后的通道数据
        processed = np.concatenate(processed_segments, axis=-1)
        return processed

    def get_dataset(self):
        return self.data

    def get_loader(self):
        return DataLoader(self.data, 
                         batch_size=self.batch_size,
                         shuffle=self.shuffle,
                         num_workers=self.num_workers)
