# 全球校园人工智能算法精英赛 (AIC) 参赛项目
# Copyright (C) 2025 六百六十(ERQI CHEN, ENZE ZHU, TIAN HAO)
# 此程序根据 GNU GPLv3 许可证发布。详见随附的 LICENSE 文件。

import paddle
import numpy as np
import xarray as xr
import pandas as pd
from paddle.io import Dataset, DataLoader
import os


class FireRiskDataset(Dataset):
    def __init__(self, nc_file1_path, nc_file2_path, npy_file_path, 
                 nc1_variables=None, nc2_variables=None, sequence_length=6, transform=None):
        super(FireRiskDataset, self).__init__()
        
        self.sequence_length = sequence_length
        self.transform = transform
        
        # 加载数据
        self.ds1 = xr.open_dataset(nc_file1_path)
        self.ds2 = xr.open_dataset(nc_file2_path)
        self.npy_data = np.load(npy_file_path)
        
        # 读取和处理时间戳
        self._load_timestamps()
        
        # 确定要使用的变量
        self.nc1_variables = self._validate_variables(nc1_variables, self.ds1, "第一个NC文件")
        self.nc2_variables = self._validate_variables(nc2_variables, self.ds2, "第二个NC文件")
        
        # 获取数据维度信息
        self.total_time_steps = len(self.timestamps)
        self.c1_channels = len(self.nc1_variables)
        self.c2_channels = len(self.nc2_variables)
        
        self._print_time_info()
    
    def _load_timestamps(self):
        """加载时间戳并转换为pd.Timestamp"""
        # 原始时间戳
        self.raw_timestamps = self.ds1['valid_time'].values
        
        # 转换为pandas Timestamp
        self.timestamps = pd.DatetimeIndex([pd.Timestamp(ts) for ts in self.raw_timestamps])
        
        print(f"时间戳类型: {type(self.timestamps[0])}")
    
    def _validate_variables(self, user_variables, dataset, dataset_name):
        """验证变量是否存在"""
        available_vars = list(dataset.data_vars.keys())
        print(f"{dataset_name}可用变量: {available_vars}")
        
        if user_variables is None:
            return available_vars
        else:
            for var in user_variables:
                if var not in available_vars:
                    raise ValueError(f"变量 '{var}' 在{dataset_name}中不存在")
            return user_variables
    
    def _print_time_info(self):
        """打印时间信息"""
        print("=== 时间信息 ===")
        print(f"时间范围: {self.timestamps[0]} 到 {self.timestamps[-1]}")
        print(f"总时间步数: {self.total_time_steps}")
        print(f"可用样本数: {self.__len__()}")
    
    def get_sequence_timestamps(self, start_index):
        """
        获取序列的时间戳信息
        
        Returns:
            dict: 包含pd.Timestamp的时间信息
        """
        end_index = start_index + self.sequence_length
        label_index = end_index
        
        return {
            'sequence_start': self.timestamps[start_index],  # pd.Timestamp
            'sequence_end': self.timestamps[end_index - 1],  # pd.Timestamp
            'label_time': self.timestamps[label_index],      # pd.Timestamp
            'sequence_dates': [self.timestamps[i] for i in range(start_index, end_index)]  # List[pd.Timestamp]
        }
    
    def __getitem__(self, index):
        """
        获取数据样本
        
        Returns:
            input_data: 输入数据
            label: 标签数据  
            time_info: 包含pd.Timestamp的字典
        """
        # 获取时间信息（pd.Timestamp类型）
        # time_info = self.get_sequence_timestamps(index)['label_time']
        
        # 计算实际的时间索引
        start_time = index
        end_time = start_time + self.sequence_length
        
        # 从第一个NC文件获取6个时间步的数据
        nc1_sequence = []
        for i in range(start_time, end_time):
            time_step_data = []
            for var_name in self.nc1_variables:
                var_data = self.ds1[var_name].isel(valid_time=i).values
                time_step_data.append(var_data)
            time_step_combined = np.stack(time_step_data, axis=0)
            nc1_sequence.append(time_step_combined)
        
        # 从第二个NC文件获取最后一个时间点的数据
        nc2_time_step_data = []
        for var_name in self.nc2_variables:
            var_data = self.ds2[var_name].isel(valid_time=end_time-1).values
            nc2_time_step_data.append(var_data)
        nc2_combined = np.stack(nc2_time_step_data, axis=0)
        
        # 从NPY文件获取下一个时间点的数据作为label  # 数据开头不放
        label = self.npy_data[start_time]
        
        # 合并输入数据
        nc1_combined = np.concatenate(nc1_sequence, axis=0)
        input_data = np.concatenate([nc1_combined, nc2_combined], axis=0)
        
        # 转换为float32
        input_data = input_data.astype('float32')
        label = label.astype('float32')
        
        if self.transform is not None:
            input_data = self.transform(input_data)
            label = self.transform(label)
        
        return input_data, label
    
    def find_samples_by_timestamp(self, target_timestamp):
        """
        根据pd.Timestamp查找样本索引
        
        Args:
            target_timestamp: pd.Timestamp对象
            
        Returns:
            list: 匹配的样本索引列表
        """
        # 找到对应的时间索引
        time_indices = np.where(self.timestamps == target_timestamp)[0]
        
        # 只返回可以作为序列起始点的索引
        valid_indices = [idx for idx in time_indices 
                        if idx + self.sequence_length < len(self.timestamps)]
        
        return valid_indices
    
    def get_timestamp_strings(self, index):
        """
        获取格式化的时间字符串
        
        Returns:
            dict: 包含格式化时间字符串的字典
        """
        time_info = self.get_sequence_timestamps(index)
        
        return {
            'sequence_start_str': time_info['sequence_start'].strftime('%Y-%m-%d %H:%M:%S'),
            'sequence_end_str': time_info['sequence_end'].strftime('%Y-%m-%d %H:%M:%S'),
            'label_time_str': time_info['label_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'sequence_dates_str': [ts.strftime('%Y-%m-%d') for ts in time_info['sequence_dates']]
        }
    
    def __len__(self):
        return self.total_time_steps - self.sequence_length
    
    def close(self):
        self.ds1.close()
        self.ds2.close()


def create_dataloader(nc_file1_path, nc_file2_path, npy_file_path, 
                           nc1_variables=None, nc2_variables=None,
                           batch_size=32, sequence_length=6, shuffle=True, 
                           num_workers=0):
    """
    创建NetCDF数据加载器
    
    Args:
        nc_file1_path: 第一个NC文件路径
        nc_file2_path: 第二个NC文件路径
        npy_file_path: NPY文件路径
        nc1_variables: 第一个NC文件中要使用的变量列表
        nc2_variables: 第二个NC文件中要使用的变量列表
        batch_size: 批次大小
        sequence_length: 序列长度
        shuffle: 是否打乱数据
        num_workers: 数据加载进程数
    
    Returns:
        DataLoader实例
    """
    
    # 创建数据集
    dataset = FireRiskDataset(
        nc_file1_path=nc_file1_path,
        nc_file2_path=nc_file2_path,
        npy_file_path=npy_file_path,
        nc1_variables=nc1_variables,
        nc2_variables=nc2_variables,
        sequence_length=sequence_length
    )
    
    
    # 创建DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True  # 丢弃最后一个不完整的batch
    )
    
    return dataloader
