# 全球校园人工智能算法精英赛 (AIC) 参赛项目
# Copyright (C) 2025 六百六十(ERQI CHEN, ENZE ZHU, TIAN HAO)
# 此程序根据 GNU GPLv3 许可证发布。详见随附的 LICENSE 文件。

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import pandas as pd

from utils.yinglong_predictor import YingLongAdaptedPredictor


class ChannelMlp(nn.Layer):
    # 对每个点进行mlp
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(ChannelMlp, self).__init__()
        
        # 1*1卷积 线性层
        self.linear1 = nn.Conv2D(in_channels = in_channels,
                                 out_channels = hidden_channels,
                                 kernel_size = 1,
                                 bias_attr = True)
        
        self.activate = F.gelu

        self.linear2 = nn.Conv2D(in_channels = hidden_channels,
                                 out_channels = out_channels,
                                 kernel_size = 1,
                                 bias_attr = True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activate(x)
        x = self.linear2(x)
        return x


def channel_layer_norm(x, eps=1e-5):
    """
    对 B*C*H*W 张量在每个 channel 上做 LayerNorm
    """
    # 计算每个 channel 的均值和方差
    mean = x.mean(axis=[2, 3], keepdim=True)  # [B, C, 1, 1]
    var = x.var(axis=[2, 3], keepdim=True)    # [B, C, 1, 1]
    
    # 归一化
    x_normalized = (x - mean) / paddle.sqrt(var + eps)
    return x_normalized


class BinarySegmentationHead(nn.Layer):
    # 二分类头
    
    def __init__(self, in_channels):
        super(BinarySegmentationHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 1, bias_attr = True)

        self.sigmoid = F.sigmoid
        
        
    def forward(self, x, hard_labels=True, threshold = 0.5):
        # 原始输出 [B, 1, H, W]
        x = self.conv(x)
        
        # 返回概率值 [0, 1]
        probabilities = self.sigmoid(x)
        if hard_labels:
            # 返回硬标签  0 / 1
            return (probabilities > threshold).float()
        else:
            # 返回概率值 [0, 1]
            return probabilities


class FireRiskYinglong(nn.Layer):
    # 预测火险
    
    def __init__(self, in_channels, adapter_channels,
                 mlp_hidden_channels,
                 cfg):
        super(FireRisk, self).__init__()

        self.adapter = ChannelMlp(in_channels = in_channels,
                                  hidden_channels = adapter_channels,
                                  out_channels = 1+24)
        self.mlp = ChannelMlp(in_channels = 1152,
                              hidden_channels = mlp_hidden_channels,
                              out_channels = in_channels)
        self.segmenthead = BinarySegmentationHead(in_channels = in_channels)
        
        self.yinglong_cfg = cfg
        self.yinglong = YingLongAdaptedPredictor(cfg)

    def forward(self, x, hard_labels = True, threshold = 0.5):
        x1 = x
        x1 = self.adapter(x1)

        # create time stamps
        cur_time = pd.Timestamp("20230401")
        time_stamps = [[cur_time]]
        for _ in range(48):
            cur_time += pd.Timedelta(hours=1)
            time_stamps.append([cur_time])

        x1 = channel_layer_norm(x1)
        x1 = x1.numpy()
        x1_d = x1.copy()[:,1:]
        x1_d_in = np.reshape(x1_d, (1,24,440,408))
        x1_d_nwp = np.repeat(x1_d_in, 48, axis=0)
        x1_g = x1.copy()[:,0].squeeze()


        x1 = self.yinglong.predict(x1_d_in, time_stamps, x1_d_nwp, x1_g)
        x1 = x1.reshape(x1, (1,1152,440,408))
        x1 = paddle.Tensor(x1)

        x1 = self.mlp(x1)
        
        x = channel_layer_norm(x) + x1
        x = self.segmenthead(x, hard_labels = hard_labels, threshold = threshold)
        return x
    

class FireRiskConv(nn.Layer):
    def __init__(self, in_channels, adapter_channels,
                 mlp_hidden_channels,
                 ):
        super(FireRiskConv, self).__init__()

        self.adapter = ChannelMlp(in_channels = in_channels,
                                  hidden_channels = adapter_channels,
                                  out_channels = 1+24)
        self.mlp = ChannelMlp(in_channels = 48,
                              hidden_channels = mlp_hidden_channels,
                              out_channels = in_channels)
        self.segmenthead = BinarySegmentationHead(in_channels = in_channels)
        
        self.nwp = nn.Sequential(
            nn.Conv2D(in_channels=25, out_channels=36, kernel_size=5, padding='same'),
            nn.Conv2D(in_channels=36, out_channels=48, kernel_size=5, padding='same')
        )

    def forward(self, x, hard_labels = True, threshold = 0.5):
        x1 = x
        x1 = self.adapter(x1)

        x1 = self.nwp(x1)

        x1 = self.mlp(x1)
        
        x = channel_layer_norm(x) + x1
        x = self.segmenthead(x, hard_labels = hard_labels, threshold = threshold)
        x = paddle.squeeze(x, axis=1)
        return x