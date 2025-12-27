# 全球校园人工智能算法精英赛 (AIC) 参赛项目
# Copyright (C) 2025 六百六十(ERQI CHEN, ENZE ZHU, TIAN HAO)
# 此程序根据 GNU GPLv3 许可证发布。详见随附的 LICENSE 文件。

import paddle
import paddle.nn as nn
import numpy as np
import xarray as xr
import pandas as pd
from paddle.io import Dataset, DataLoader
import os

from utils.dataloader import create_dataloader
from utils.data_split_musk import generate_binary_sequence
from utils.metrics import PixelBinaryFocalLoss, pixel_binary_f1_score, pixel_binary_accuracy


def train_model(model,
                times,
                epochs,
                lr_scheduler,
                data_paths,
                val_ratio = 0.15,
                batch_out = True,
                lr_shedular_type = 'batch'):
    
    # 文件路径  # 创造性地使用了最麻烦的写法
    nc_file1 = data_paths['land']
    nc_file2 = data_paths['pressure_levels']
    npy_file = data_paths['label']
    
    # 指定要使用的变量
    nc1_vars = ['lai_lv', 'lai_hv', 'sp', 'v10', 'u10', 'skt', 'd2m', 'precipitation']
    nc2_vars = ['v_850', 'v_500', 'u_850', 'u_500', 't_850', 't_500', 'q_850', 'q_500', 'z_850', 'z_500']


    # ************* 一键开启 自动调优 ***************
    config = {
        "kernel": {
            "enable": True,   # 开启 kernel 调优
            "tuning_range": [1, 3],
        },
        "layout": {
            "enable": True,  # 开启 layout 调优
        }
    }
    paddle.incubate.autotune.set_config(config) # 添加配置
    # ************* 一键开启 自动调优 ***************


    # 创建数据加载器
    data_loader = create_dataloader(
        nc_file1_path=nc_file1,
        nc_file2_path=nc_file2,
        npy_file_path=npy_file,
        nc1_variables=nc1_vars,    # 指定第一个文件的变量
        nc2_variables=nc2_vars,    # 指定第二个文件的变量
        batch_size=1,
        sequence_length=6,
        shuffle=True,
        num_workers=0
    )

    len_data = len(data_loader)
    len_val = int(val_ratio*len_data)
    len_train = len_data - len_val
    data_split_musk = generate_binary_sequence(len_data, val_ratio)


    # 损失函数和优化器
    criterion = PixelBinaryFocalLoss()
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=lr_scheduler
    )
    

    # 训练循环
    model.train()
    for epoch in range(1, epochs+1):
        model.train()
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            loss = 0.0

            if data_split_musk[batch_idx] == 1:
            
                # 前向传播
                outputs = model(inputs, hard_labels = False)
                loss = criterion(outputs, labels)
            
                # 反向传播
                optimizer.clear_gradients()
                loss.backward()
                optimizer.step()

                if batch_out:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss:.8f}')
            
            if not isinstance(lr_scheduler,float):
                if lr_shedular_type == 'batch':
                    lr_scheduler.step()


        if not isinstance(lr_scheduler,float):
            if lr_shedular_type == 'epoch':
                lr_scheduler.step()            
        
        print(f'Epoch {epoch}/{epochs} completed.')

        model.eval()
        total_train_loss = 0.
        total_train_acc = 0.
        total_val_loss = 0.
        total_val_acc = 0.
        for batch_idx, (inputs, labels) in enumerate(data_loader):
            # 前向传播
            outputs = model(inputs, hard_labels = False)
            loss = criterion(outputs, labels)
            
            if data_split_musk[batch_idx] == 1:
                total_train_loss += loss.item()
                total_train_acc += pixel_binary_accuracy(outputs, labels, threshold=0.5)
                total_train_f1 += pixel_binary_f1_score(outputs, labels, threshold=0.5)

            elif data_split_musk[batch_idx] == 0:
                total_val_loss += loss.item()
                total_val_acc += pixel_binary_accuracy(outputs, labels, threshold=0.5)
                total_val_f1 += pixel_binary_f1_score(outputs, labels, threshold=0.5)
            
        avg_train_loss = total_train_loss / len_train
        avg_train_acc = total_train_acc / len_train
        
        avg_val_loss = total_val_loss / len_val
        avg_val_acc = total_val_acc / len_val

        print(f'Average train_loss: {avg_train_loss:.8f}    Average train_accuracy: {avg_train_acc:.8f}')
        print(f'Average val_loss: {avg_val_loss:.8f}    Average val_accuracy: {avg_val_acc:.8f}')
        
        # 每个epoch结束后可以保存模型
        if epoch % 10 == 0:
            paddle.save(model.state_dict(), f'YOUR/PATH/model_checkpoints/model_{times}_epoch_{epoch}.pdparams')

    if epochs % 10 != 0:
        paddle.save(model.state_dict(), f'YOUR/PATH/model_checkpoints/model_{times}_epoch_{epochs}.pdparams')
    
    return model
