# 全球校园人工智能算法精英赛 (AIC) 参赛项目
# Copyright (C) 2025 六百六十(ERQI CHEN, ENZE ZHU, TIAN HAO)
# 此程序根据 GNU GPLv3 许可证发布。详见随附的 LICENSE 文件。

import numpy as np

def generate_binary_sequence(length, zero_ratio):
    """
    生成随机{0,1}序列
    
    参数:
    length: 序列长度
    zero_ratio: 0的比例 (0到1之间的小数)
    
    返回:
    list: 包含0和1的列表
    """
    if not 0 <= zero_ratio <= 1:
        raise ValueError("zero_ratio必须在0到1之间")
    
    # 计算0和1的数量
    zeros_count = int(length * zero_ratio)
    ones_count = length - zeros_count
    
    # 创建序列
    sequence = [0] * zeros_count + [1] * ones_count
    
    # 随机打乱顺序
    np.random.shuffle(sequence)
    
    return sequence
