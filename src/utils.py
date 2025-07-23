import numpy as np


def train_test_divide(ori_data, generated_data, ori_time, generated_time):
    """划分训练集和测试集"""
    # 随机打乱索引
    no = len(ori_data)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no*0.8)]
    test_idx = idx[int(no*0.8):]
    
    # 划分数据
    train_x = ori_data[train_idx]
    test_x = ori_data[test_idx]
    train_x_hat = generated_data[train_idx]
    test_x_hat = generated_data[test_idx]
    
    # 划分时间
    train_t = ori_time[train_idx]
    test_t = ori_time[test_idx]
    train_t_hat = generated_time[train_idx]
    test_t_hat = generated_time[test_idx]
    
    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat

def extract_time(data):
    """提取时间信息"""
    time = np.array([len(seq) for seq in data])
    max_seq_len = np.max(time)
    return time, max_seq_len

def reshape_sequence_data(data):
    """将3D时序数据转换为2D矩阵"""
    if len(data.shape) != 3:
        raise ValueError("输入数据必须是3维数组 (样本数×时间步×特征数)")
    return data.reshape(data.shape[0], -1)