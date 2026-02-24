"""
Custom Data Loader for Weather Prediction
完全对齐 LSTM 实验的数据加载方式
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class Dataset_Custom(Dataset):
    """
    自定义气象数据集
    """
    def __init__(self, root_path, data_path, flag='train', size=None,
                 features='MS', target='temperature', 
                 scale=True, inverse=False):
        """
        参数说明：
        - root_path: 数据根目录
        - data_path: 数据文件名（如 'city_beijing.csv'）
        - flag: 'train' / 'val' / 'test'
        - size: [seq_len, label_len, pred_len] 
                对应 [window_size, 0, horizon]
        - features: 'MS' (多变量预测单变量) / 'M' (多变量预测多变量)
        - target: 目标变量名（默认 'temperature'）
        - scale: 是否归一化（你的数据已归一化，但我们保持接口一致）
        - inverse: 预测时是否需要反归一化
        """
        # 尺寸设置
        if size is None:
            self.seq_len = 96   # 默认输入窗口
            self.label_len = 0   # Informer 中间辅助长度，我们设为0
            self.pred_len = 1    # 默认预测长度
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # 数据集划分标志
        assert flag in ['train', 'val', 'test']
        self.set_type = flag
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        
        self.root_path = root_path
        self.data_path = data_path
        
        # 读取数据
        self.__read_data__()

    def __read_data__(self):
        """读取并划分数据"""
        # 读取完整数据
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 假设数据格式：第一列为时间，最后一列为 temperature，中间为特征
        
        # 获取列名
        cols = list(df_raw.columns)
        
        # 移除时间列（假设第一列是时间）
        if 'date' in cols[0].lower() or 'time' in cols[0].lower():
            cols.remove(cols[0])
            df_raw = df_raw[cols]
        
        # 确保 temperature 是最后一列（如果不是，需要调整）
        if self.target not in cols:
            raise ValueError(f"目标变量 '{self.target}' 不在数据列中！")

        self.target_idx = cols.index(self.target)
        
        # 数据集划分（按 LSTM 的方式）
        n = len(df_raw)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        border1s = {
            'train': 0,
            'val': train_end - self.seq_len,
            'test': val_end - self.seq_len
        }
        border2s = {
            'train': train_end,
            'val': val_end,
            'test': n
        }
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 选择特征列
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns  # 所有特征
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 单变量预测（仅使用 temperature）
            df_data = df_raw[[self.target]]
        
        # 归一化
        if self.scale:
            train_data = df_data[border1s['train']:border2s['train']]
            self.scaler = StandardScaler()
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # 提取目标变量索引
        self.target_idx = df_data.columns.get_loc(self.target)
        
        # 保存当前数据集的数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
    def __getitem__(self, index):
        """
        返回样本
        对齐 LSTM 的滑动窗口方式
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Informer 需要的格式
        # seq_x: [seq_len, feature_dim]
        # seq_y: [pred_len, feature_dim] (训练时) 或 [pred_len, 1] (预测时)
        
        if self.features == 'MS':
            # 多变量输入，单变量输出
            seq_y = seq_y[:, self.target_idx:self.target_idx+1]
        
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """反归一化（仅对 temperature）"""
        if self.scale:
            # 创建全零数组
            temp_data = np.zeros((data.shape[0], len(self.scaler.mean_)))
            temp_data[:, self.target_idx] = data[:, 0]
            # 反归一化
            temp_data = self.scaler.inverse_transform(temp_data)
            return temp_data[:, self.target_idx:self.target_idx+1]
        else:
            return data


def data_provider(args, flag):
    """
    数据提供器（对接官方接口）
    """
    Data = Dataset_Custom
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        scale=True,
        inverse=False
    )
    
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last
    )
    
    return data_set, data_loader