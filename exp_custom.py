import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from data_loader_custom import data_provider
from models.model import Informer
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

warnings.filterwarnings('ignore')


class Exp_Custom(object):
    """
    自定义实验类
    """
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        """设备选择"""
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        """构建 Informer 模型"""
        model = Informer(
            enc_in=self.args.enc_in,
            dec_in=self.args.dec_in,
            c_out=self.args.c_out,
            seq_len=self.args.seq_len,
            label_len=self.args.label_len,
            out_len=self.args.pred_len,
            factor=self.args.factor,
            d_model=self.args.d_model,
            n_heads=self.args.n_heads,
            e_layers=self.args.e_layers,
            d_layers=self.args.d_layers,
            d_ff=self.args.d_ff,
            dropout=self.args.dropout,
            attn=self.args.attn,
            embed=self.args.embed,
            freq=self.args.freq,
            activation=self.args.activation,
            output_attention=self.args.output_attention,
            distil=self.args.distil,
            mix=self.args.mix,
            device=self.device
        )
        return model

    def _get_data(self, flag):
        """获取数据"""
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """优化器选择（对齐 LSTM: Adam, lr=1e-4）"""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, task_type):
        """
        损失函数选择
        对齐 LSTM 的损失设计
        """
        if task_type == 'single_point':
            # 单点预测：MSE
            criterion = nn.MSELoss()
        elif task_type == 'sequence':
            # 序列预测：加权 MSE
            criterion = WeightedMSELoss(pred_len=self.args.pred_len)
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        """
        验证阶段
        对齐 LSTM 的验证流程
        """
        self.model.eval()
        total_loss = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                
                # Informer 需要 decoder input
                # 我们使用零填充（类似 teacher forcing 的起点）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                
                # 前向传播
                outputs = self.model(batch_x, None, dec_inp, None)
                
                # 提取预测值
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                
                loss = criterion(pred, true)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        """
        训练流程
        完全对齐 LSTM 的训练策略
        """
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.task_type)

        # 训练日志
        train_losses = []
        vali_losses = []
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Decoder input（使用零填充）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向传播
                outputs = self.model(batch_x, None, dec_inp, None)
                
                # 计算损失
                pred = outputs[:, -self.args.pred_len:, :]
                true = batch_y[:, -self.args.pred_len:, :]
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                # 反向传播
                loss.backward()
                model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            
            train_losses.append(train_loss)
            vali_losses.append(vali_loss)
            
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            
            # Early Stopping
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 保存训练日志
        np.save(os.path.join(path, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(path, 'vali_losses.npy'), np.array(vali_losses))
        
        # 加载最优模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def test(self, setting):
        """
        测试流程
        对齐 LSTM 的测试和评估方式
        """
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        # 记录推理时间
        inference_start = time.time()
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # Decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 推理
                outputs = self.model(batch_x, None, dec_inp, None)
                
                pred = outputs[:, -self.args.pred_len:, :].detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)

        inference_time = time.time() - inference_start
        
        # 合并结果
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 反归一化（仅 temperature）
        preds = test_data.inverse_transform(preds.reshape(-1, 1))
        trues = test_data.inverse_transform(trues.reshape(-1, 1))
        
        # 重新调整形状
        if self.args.task_type == 'single_point':
            preds = preds.reshape(-1)
            trues = trues.reshape(-1)
        else:
            preds = preds.reshape(-1, self.args.pred_len)
            trues = trues.reshape(-1, self.args.pred_len)
        
        # 计算指标
        mae, rmse, r2 = metric(preds, trues)
        
        print(f'Test Results:')
        print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
        print(f'Inference Time: {inference_time:.2f}s')
        
        # 保存结果
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path + 'metrics.npy', np.array([mae, rmse, r2, inference_time]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        
        return mae, rmse, r2, inference_time


class WeightedMSELoss(nn.Module):
    """
    加权 MSE 损失
    对齐 LSTM 的序列预测损失设计
    """
    def __init__(self, pred_len):
        super(WeightedMSELoss, self).__init__()
        # 指数衰减权重
        weights = np.exp(-0.1 * np.arange(pred_len))
        weights = weights / weights.sum()
        self.weights = torch.FloatTensor(weights)
    
    def forward(self, pred, true):
        """
        pred: [B, pred_len, 1]
        true: [B, pred_len, 1]
        """
        self.weights = self.weights.to(pred.device)
        mse = (pred - true) ** 2
        mse = mse.squeeze(-1)  # [B, pred_len]
        weighted_mse = (mse * self.weights).mean()
        return weighted_mse