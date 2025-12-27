# YingLong: 数据驱动的区域天气预测模型

本文件夹需要包含YingLong模型的实现文件。这是一个基于人工智能的区域天气预测模型。YingLong采用全局和局部块的并行结构来捕捉多尺度气象特征，在预测性能上与WRF-ARW等动力区域模型相当，但运行速度显著更快。

## 模型架构

<div align=center>
    <img src="doc/fig_arch1.jpeg" width="70%" height="auto" >
</div>

YingLong专门设计用于美国东南部地区（80-110W, 30-42N）的区域天气预测，采用Lambert投影，网格点为440 × 408。

## 安装要求

### 1. 安装PaddlePaddle

见PaddleScience官方开源文件。


## 模型文件下载

由于模型文件（`.pdmodel`, `.pdiparams`, `.pdiparams.info`）体积较大，请前往PaddleScience官方仓库单独下载。

**所需文件结构（示例）：**
```
inference/
├── yinglong_eastern.pdmodel
├── yinglong_eastern.pdiparams
└── yinglong_eastern.pdiparams.info
```


## 重要说明

1. **模型文件要求：** 推理模型文件（`.pdmodel`, `.pdiparams`, `.pdiparams.info`）是运行预测所必需的，由于文件较大需要单独下载。

2. **硬件要求：**
   - 推荐使用支持CUDA的GPU
   - 足够的磁盘空间用于存储模型文件和输出数据

3. **商业使用：** 根据上海张江数学研究所和百度的声明，YingLong模型严禁用于商业用途。


## 引用

如果您在研究中使用此代码，请引用原始的YingLong论文。

## 许可证

YingLong was released by Shanghai Zhangjiang Institute of Mathematics, Baidu inc.

The commercial use of these models is forbidden.