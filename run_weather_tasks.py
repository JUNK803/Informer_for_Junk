import argparse
import os
import torch
import numpy as np
import time
from exp_custom import Exp_Custom


# 任务定义（对齐 LSTM）
TASKS = [
    (24, 1, "single_point"),
    (96, 1, "single_point"),
    (24, 6, "single_point"),
    (96, 6, "single_point"),
    (96, 6, "sequence"),
    (96, 24, "single_point"),
    (96, 24, "sequence"),
]


def get_args(window_size, horizon, task_type, data_path):
    """
    生成实验参数
    完全对齐 LSTM 的超参数设置
    """
    parser = argparse.ArgumentParser(description='Informer for Weather Prediction')

    # 基本配置
    parser.add_argument('--model', type=str, default='informer', help='model name')
    parser.add_argument('--data', type=str, default='custom', help='data type')
    parser.add_argument('--root_path', type=str, default='./data/', help='root path of data')
    parser.add_argument('--data_path', type=str, default=data_path, help='data file')
    parser.add_argument('--features', type=str, default='MS', help='M: multivariate, MS: multivariate predict univariate, S: univariate')
    parser.add_argument('--target', type=str, default='temperature', help='target feature')
    parser.add_argument('--freq', type=str, default='h', help='h: hourly')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoints directory')

    # 数据尺寸
    parser.add_argument('--seq_len', type=int, default=window_size, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length (设为0)')
    parser.add_argument('--pred_len', type=int, default=horizon, help='prediction sequence length')
    
    # 模型参数（对齐 LSTM 的复杂度）
    parser.add_argument('--enc_in', type=int, default=92, help='encoder input size (特征数)')   # 实际检查特征数为92
    parser.add_argument('--dec_in', type=int, default=92, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size (预测 temperature)')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model (对齐 LSTM hidden_size)')
    parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers (对齐 LSTM num_layers)')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
    parser.add_argument('--padding', type=int, default=0, help='padding type')
    parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder', default=True)
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout (对齐 LSTM)')
    parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)

    # 训练策略（完全对齐 LSTM）
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs (对齐 LSTM)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size (对齐 LSTM)')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience (对齐 LSTM)')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate (对齐 LSTM)')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    # 任务类型（自定义字段）
    parser.add_argument('--task_type', type=str, default=task_type, help='single_point or sequence')

    args = parser.parse_args([])  # 空列表，使用默认值
    
    # 检查 GPU 可用性
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    return args


def run_single_task(window_size, horizon, task_type, data_path, city_name):
    """
    运行单个任务
    """
    print(f"\n{'='*80}")
    print(f"Running Task: Window={window_size}, Horizon={horizon}, Type={task_type}")
    print(f"City: {city_name}")
    print(f"{'='*80}\n")
    
    # 获取参数
    args = get_args(window_size, horizon, task_type, data_path)
    
    # 设置随机种子（对齐 LSTM）
    fix_seed = 2021
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 创建实验
    setting = f'{city_name}_informer_w{window_size}_h{horizon}_{task_type}'
    exp = Exp_Custom(args)
    
    # 记录训练时间
    train_start = time.time()
    
    # 训练
    print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train(setting)
    train_time = time.time() - train_start
    
    # 测试
    print('>>>>>>>testing >>>>>>>>>>>>>>>>>>>>>>>>>>')
    mae, rmse, r2, inference_time = exp.test(setting)
    
    print(f'\nTask Completed!')
    print(f'Training Time: {train_time:.2f}s')
    print(f'Inference Time: {inference_time:.2f}s')
    
    return {
        'window': window_size,
        'horizon': horizon,
        'task_type': task_type,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_time': train_time,
        'inference_time': inference_time
    }


def run_all_tasks_for_city(data_path, city_name):
    """
    为单个城市运行所有 7 个任务
    """
    results = []
    
    for window_size, horizon, task_type in TASKS:
        result = run_single_task(window_size, horizon, task_type, data_path, city_name)
        results.append(result)
    
    # 保存汇总结果
    results_dir = './results_summary/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 生成 Markdown 报告
    report_path = os.path.join(results_dir, f'{city_name}_informer_results.md')
    generate_report(results, city_name, report_path)
    
    print(f"\n{'='*80}")
    print(f"All tasks completed for {city_name}!")
    print(f"Results saved to: {report_path}")
    print(f"{'='*80}\n")
    
    return results


def generate_report(results, city_name, output_path):
    """
    生成 Markdown 格式的结果报告
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Informer Results - {city_name}\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Task | Window | Horizon | Type | MAE | RMSE | R² | Train Time (s) | Inference Time (s) |\n")
        f.write("|------|--------|---------|------|-----|------|-------|----------------|-------------------|\n")
        
        for i, res in enumerate(results, 1):
            f.write(f"| {i} | {res['window']} | {res['horizon']} | {res['task_type']} | "
                   f"{res['mae']:.4f} | {res['rmse']:.4f} | {res['r2']:.4f} | "
                   f"{res['train_time']:.2f} | {res['inference_time']:.2f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        for i, res in enumerate(results, 1):
            f.write(f"### Task {i}: Window={res['window']}, Horizon={res['horizon']}, Type={res['task_type']}\n\n")
            f.write(f"- **MAE**: {res['mae']:.6f}\n")
            f.write(f"- **RMSE**: {res['rmse']:.6f}\n")
            f.write(f"- **R²**: {res['r2']:.6f}\n")
            f.write(f"- **Training Time**: {res['train_time']:.2f}s\n")
            f.write(f"- **Inference Time**: {res['inference_time']:.2f}s\n\n")


if __name__ == '__main__':
    """
    使用示例：
    
    python run_weather_tasks.py
    
    你需要修改下面的参数：
    - data_path: 你的数据文件名
    - city_name: 城市名称
    """
    
    # ============== 配置区域 ==============
    data_path = 'Albuquerque_wide.csv'  # 修改为你的数据文件名
    city_name = 'Albuquerque_wide'            # 修改为城市名称
    # =====================================
    
    # 运行所有任务
    results = run_all_tasks_for_city(data_path, city_name)
    
    print("\n🎉 All tasks completed successfully!")