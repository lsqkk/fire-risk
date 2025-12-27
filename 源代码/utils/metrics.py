# 全球校园人工智能算法精英赛 (AIC) 参赛项目
# Copyright (C) 2025 六百六十(ERQI CHEN, ENZE ZHU, TIAN HAO)
# 此程序根据 GNU GPLv3 许可证发布。详见随附的 LICENSE 文件。

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class PixelBinaryCrossEntropyLoss(nn.Layer):
    def __init__(self, weight=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits, label):
        """
        计算二值交叉熵损失
        Args:
            logits (Tensor): 网络原始输出 [N, 1, H, W]
            label (Tensor): 真实标签 [N, H, W]，取值为0或1
        Returns:
            Tensor: 计算得到的损失值
        """
        # 如果logits是[N, 1, H, W]，先压缩为[N, H, W]
        if len(logits.shape) == 4 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        
        # 创建掩码，忽略特定索引
        mask = (label != self.ignore_index)
        mask = mask.astype('float32')
        
        # 计算sigmoid激活后的概率
        prob = F.sigmoid(logits)
        
        # 计算二值交叉熵[citation:10]
        loss = F.binary_cross_entropy(prob, label.astype('float32'), 
                                    reduction='none')
        
        # 应用掩码
        loss = loss * mask
        
        # 应用权重
        if self.weight is not None:
            loss = loss * self.weight
        
        #  reduction处理[citation:4]
        if self.reduction == 'mean':
            if mask.sum() > 0:
                loss = loss.sum() / mask.sum()
            else:
                loss = loss.mean() * 0
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        return loss

def pixel_binary_accuracy(pred, label, threshold=0.5, ignore_index=255):
    """
    计算像素级二分类准确率
    Args:
        pred (Tensor): 网络输出 [N, 1, H, W] 或 [N, H, W]
        label (Tensor): 真实标签 [N, H, W]
        threshold (float): 二值化阈值
        ignore_index (int): 需要忽略的标签值
    Returns:
        Tensor: 准确率值
    """
    # 调整pred形状为[N, H, W]
    if len(pred.shape) == 4:
        pred = pred.squeeze(1)
    
    # 将logits通过sigmoid转换为概率，然后二值化
    prob = F.sigmoid(pred)
    binary_pred = (prob > threshold).astype('int64')
    
    # 创建有效掩码（忽略ignore_index的区域）
    valid_mask = (label != ignore_index)
    
    # 计算正确预测的像素
    correct = (binary_pred == label.astype('int64')) & valid_mask
    
    # 计算准确率：正确像素数 / 有效像素总数[citation:5]
    total_valid = valid_mask.astype('float32').sum()
    if total_valid > 0:
        acc = correct.astype('float32').sum() / total_valid
    else:
        acc = paddle.to_tensor(0.0)
    
    return acc