# 基于机器学习的火灾风险预测

一个基于机器学习的火灾风险预测系统，包含预处理数据和模型训练。

## 项目概述

提出并实现了一种 **数据与模型双驱动的智慧气象山火防控框架**，旨在通过融合多源气象数据与深度学习模型，实现对山火发生概率的高精度、高时空分辨率预测。具备较强数值模拟能力，并专门针对火险因子进行建模，更加适应中国复杂地形与气候条件。

## 技术特色

- **双驱动架构**：数据驱动 + 物理机制融合  
- **高分辨率插值**：采用克里金插值法将气象数据从0.25°提升至0.01°  
- **轻量化模型**：基于PaddlePaddle的轻量卷积网络，兼顾性能与可解释性  
- **多源数据融合**：整合ERA5再分析数据、NASA火点监测数据、植被指数等多类数据  
- **端到端流水线**：从数据获取、预处理、模型训练到预警输出的完整闭环系统  

## 项目结构

```
项目根目录/
├── 数据/                           # 数据集文件夹
│   ├── fire_risk_dataset/          # 火灾风险数据集（含标准化特征变量）
│   ├── model_checkpoints/          # 预训练模型权重
│   ├── yinglong_inference/         # 应龙气象大模型推理相关文件
├── 源代码/                         # 主要源代码目录
│   ├── utils/                      # 数据预处理、插值、特征工程工具
│   ├── model.py                    # 双驱动模型架构定义（适配器+CNN+MLP+残差连接）
│   ├── train.py                    # 训练脚本，支持超参数调节与模型保存
│   ├── train.ipynb                 # Jupyter Notebook交互式训练与可视化
├── requirements.txt                # Python环境依赖
└── README.md                       # 项目说明
```

## 技术架构详解

### 1. 数据来源与预处理
- **气象数据**：ERA5再分析数据（单层与日统计数据集），包括位势高度、比湿、温度、风场、降水等，详见[ERA5数据集处理说明](数据/fire_risk_dataset/README.md)。
- **火点数据**：NASA FIRMS系统提供的MODIS/VIIRS近实时全球火点监测数据
- **植被数据**：叶面积指数、高低植被覆盖度等
- **预处理流程**：
  - 数据完整性验证（xarray）
  - 物理合理性检验
  - 时空一致性分析
  - 克里金插值（降水）与双线性插值（温度、湿度、风场）
  - 特征筛选（皮尔逊相关系数）

### 2. 模型架构
采用 **“适配器-大模型-残差精修”混合架构**：
- **适配器**：1×1卷积构建的MLP，进行特征升维与非线性变换
- **主干网络**：轻量级CNN + MLP，提取时空气象特征
- **残差连接**：保留原始观测信息，增强对火险异常信号的捕捉
- **输出层**：全连接 + Sigmoid，输出火险概率图

### 3. 训练与优化
- **框架**：PaddlePaddle + PaddleScience
- **输入**：6天地面数据 + 1天高空数据（500hPa、850hPa）
- **输出**：未来时段火险概率空间分布图
- **优化策略**：轻量化卷积、参数共享、LayerNorm标准化、动态阈值划分

## 模型性能

- **准确率**：>= 99.97%
- **最低交叉熵损失**：0.01157
- **支持输出**：空间网格化火险等级图（未来特定时段）

## 快速开始

### 环境安装
```bash
pip install -r requirements.txt
```

### 数据准备
1. 下载ERA5数据（可通过CDS API或IDM工具）
2. 下载NASA FIRMS火点数据
3. 运行预处理脚本（位于`源代码/utils/`）

### 训练模型
```bash
python 源代码/train.py --epochs 50 --batch_size 32
```

或使用Jupyter Notebook交互式训练：
```bash
jupyter notebook 源代码/train.ipynb
```

## 相关资源

- [ERA5数据说明](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5)
- [NASA FIRMS火点数据](https://firms.modaps.eosdis.nasa.gov/)
- [PaddlePaddle官网](https://www.paddlepaddle.org.cn/)
- [PaddleScience科学计算库](https://github.com/PaddlePaddle/PaddleScience)


## 开源许可

全球校园人工智能算法精英赛 (AIC) 参赛项目
Copyright (C) 2025 六百六十(ERQI CHEN, ENZE ZHU, TIAN HAO)
此程序根据 GNU GPLv3 许可证发布。详见随附的 LICENSE 文件。