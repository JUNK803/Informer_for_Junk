import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def RSE(pred, true):
    """相对平方误差"""
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """相关系数"""
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    """平均绝对误差"""
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """均方误差"""
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """均方根误差"""
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """平均绝对百分比误差"""
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """均方百分比误差"""
    return np.mean(np.square((pred - true) / true))


def R2(pred, true):
    """决定系数 R²"""
    return r2_score(true.flatten(), pred.flatten())


def metric(pred, true):
    """
    计算主要评估指标
    对齐 LSTM: MAE, RMSE, R²
    
    参数：
        pred: 预测值 [N, ] 或 [N, H]
        true: 真实值 [N, ] 或 [N, H]
    
    返回：
        mae, rmse, r2
    """
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    r2 = R2(pred, true)
    
    return mae, rmse, r2


def metric_multistep(pred, true):
    """
    多步预测的详细指标
    计算每个预测步长的 MAE
    
    参数：
        pred: [N, H]
        true: [N, H]
    
    返回：
        overall_mae, step_wise_mae
    """
    overall_mae = MAE(pred, true)
    
    # 每个步长的 MAE
    step_wise_mae = []
    for h in range(pred.shape[1]):
        step_mae = MAE(pred[:, h], true[:, h])
        step_wise_mae.append(step_mae)
    
    return overall_mae, np.array(step_wise_mae)