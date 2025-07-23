# Traj-timeGAN: Trajectory Data Generation and Evaluation Framework

Traj-timeGAN is a GAN (Generative Adversarial Network) based trajectory data generation model. This framework includes complete workflows for model training, data generation, visualization, and evaluation, supporting processing and analysis of various time-series datasets.

ATTENTION: This is an undergraduate graduation project, and the results are average, but there is a popular saying in China: there aren’t many audience members in life. Therefore, I uploaded my humble work to my personal repository.

## Project Overview

Traj-timeGAN combines the advantages of time-series modeling and generative adversarial networks, enabling the generation of synthetic data with distributions similar to real trajectory data. Key features of this project include:

- Support for multiple time-series datasets (HGV vehicle trajectory data, sine curve data, etc.)
- Complete workflows for model training, data generation, and visualization
- Built-in multiple evaluation metrics to quantify generated data quality
- PCA and t-SNE for comparative visualization of data distributions

## Environment Requirements

The project depends on the following Python libraries:

```
numpy
scikit-learn
scikit-image
torch
matplotlib
```

## Installation Steps

1. Clone this project to your local machine
```bash
git clone <project repository URL>
cd Traj-timeGAN
```

2. Install dependency packages
```bash
pip install -r requirements.txt
```

## Dataset Preparation

Two types of datasets are supported:
- HGV vehicle trajectory data: filenames starting with `hgv_`
- Sine curve data: filenames starting with `sine_`

Datasets should be stored in `.npz` format, containing a `data` key with dimensions `(number of samples × time steps × number of features)`. Place datasets in the `data/` directory.

## Usage

Control the entire workflow through the `src/mainGAN.py` script, which supports four operation modes:

- `train`: Train the model only
- `generate`: Generate synthetic data only (requires pre-trained model)
- `visualize`: Visualize real and synthetic data and calculate evaluation metrics
- `all`: Execute training, generation, and visualization sequentially (default mode)

### Basic Command Format

```bash
python src/mainGAN.py --dataset <dataset name> [--mode <operation mode>]
```

### Examples

1. Complete workflow (training + generation + visualization)
```bash
python src/mainGAN.py --dataset hgv_trajectories_1 --mode all
```

2. Train model only
```bash
python src/mainGAN.py --dataset sine_subsampled_train_perc_2 --mode train
```

3. Generate synthetic data
```bash
python src/mainGAN.py --dataset hgv_trajectories_1 --mode generate
```

4. Data visualization and evaluation
```bash
python src/mainGAN.py --dataset sine_subsampled_train_perc_2 --mode visualize
```

## Project Structure

```
Traj-timeGAN/
├── data/                 # Dataset directory
├── outputs/              # Output results directory
│   └── <dataset name>/
│       ├── generated_data/  # Generated synthetic data
│       ├── models/          # Trained models
│       └── visualization/   # Visualization results and evaluation metrics
├── src/
│   ├── data_loaders/     # Data loaders
│   │   └── data_loader.py  # Data loading and preprocessing
│   ├── models/           # Model definitions
│   │   └── traj_timegan.py # GAN model structure
│   ├── mainGAN.py        # Main control program
│   ├── metrics.py        # Evaluation metrics implementation
│   ├── train.py          # Model training logic
│   └── utils.py          # Utility functions
├── requirements.txt      # Dependencies list
└── tsne.py               # Additional t-SNE visualization script
```

## Model Architecture

- **Generator**: Based on a bidirectional LSTM network, generates time-series data from random noise
- **Discriminator**: Uses 1D convolutional networks to distinguish between real and generated data
- **Training Strategy**: Standard GAN adversarial training, alternately optimizing generator and discriminator

## Evaluation Metrics

The system provides multiple quantitative evaluation metrics. Evaluation results are saved in `outputs/<dataset name>/visualization/evaluation_results.txt`:

- **Jensen-Shannon Divergence**: Measures similarity between real and generated data distributions
- **Predictive Score**: Evaluates temporal prediction capability of generated data
- **Discriminative Score**: Measures realism of generated data (lower is better)
- **Feature Variance Difference**: Compares feature variances between real and generated data

## Visualization Results

Visualization results are saved in the `outputs/<dataset name>/visualization/` directory:

- `pca_tsne.png`: Contains both PCA and t-SNE dimensionality reduction visualizations, comparing distributions of real data (blue) and generated data (red)

## Extension and Customization

1. Adding new dataset types:
   - Implement new dataset loaders in `data_loaders/data_loader.py`
   - Inherit `BaseDataLoader` and implement the `_preprocess` method

2. Adjusting model parameters:
   - Generator parameters can be modified in the `Generator` class in `models/traj_timegan.py`
   - Discriminator parameters can be modified in the `Discriminator` class in `models/traj_timegan.py`
   - Training parameters (batch size, learning rate, etc.) can be adjusted in the `TrajTimeGANTrainer` class in `src/train.py`

3. Adding new evaluation metrics:
   - Add new evaluation methods in the `MetricEvaluator` class in `src/metrics.py`
   - Integrate new metrics in the `evaluate_all` method


# Traj-timeGAN: 轨迹数据生成与评估

Traj-timeGAN是一个基于GAN（生成对抗网络）的轨迹数据生成模型。
这是一个本科毕业设计，生成效果一般，但反正人生没有多少观众，故将拙作上传至个人仓库。

## 项目简介

Traj-timeGAN结合了时间序列建模与生成对抗网络的优势，能够生成与真实轨迹数据分布相似的合成数据。该项目主要特点包括：

- 支持多种时序数据集（HGV车辆轨迹数据、正弦曲线数据等）
- 提供完整的模型训练、数据生成、可视化流程
- 内置多种评估指标，量化生成数据质量
- 采用PCA和t-SNE进行数据分布可视化对比

## 环境要求

项目依赖以下Python库：

```
numpy
scikit-learn
scikit-image
torch
matplotlib
```

## 安装步骤

1. 克隆本项目到本地
```bash
git clone <项目仓库地址>
cd Traj-timeGAN
```

2. 安装依赖包
```bash
pip install -r requirements.txt
```

## 数据集准备

支持两种类型的数据集：
- HGV轨迹数据：文件名以`hgv_`开头
- 正弦曲线数据：文件名以`sine_`开头

数据集需存储为`.npz`格式，包含`data`键，数据维度为`(样本数×时间步×特征数)`。将数据集放置在`data/`目录下。

## 使用方法

通过`src/mainGAN.py`脚本控制整个流程，支持四种运行模式：

- `train`：仅训练模型
- `generate`：仅生成合成数据（需先训练模型）
- `visualize`：可视化真实数据与合成数据并计算评估指标
- `all`：依次执行训练、生成、可视化（默认模式）

### 基本命令格式

```bash
python src/mainGAN.py --dataset <数据集名称> [--mode <运行模式>]
```

### 示例

1. 完整流程（训练+生成+可视化）
```bash
python src/mainGAN.py --dataset hgv_trajectories_1 --mode all
```

2. 仅训练模型
```bash
python src/mainGAN.py --dataset sine_subsampled_train_perc_2 --mode train
```

3. 生成合成数据
```bash
python src/mainGAN.py --dataset hgv_trajectories_1 --mode generate
```

4. 数据可视化与评估
```bash
python src/mainGAN.py --dataset sine_subsampled_train_perc_2 --mode visualize
```

## 项目结构

```
Traj-timeGAN/
├── data/                 # 数据集目录
├── outputs/              # 输出结果目录
│   └── <数据集名称>/
│       ├── generated_data/  # 生成的合成数据
│       ├── models/          # 训练好的模型
│       └── visualization/   # 可视化结果与评估指标
├── src/
│   ├── data_loaders/     # 数据加载器
│   │   └── data_loader.py  # 数据加载与预处理
│   ├── models/           # 模型定义
│   │   └── traj_timegan.py # GAN模型结构
│   ├── mainGAN.py        # 主控制程序
│   ├── metrics.py        # 评估指标实现
│   ├── train.py          # 模型训练逻辑
│   └── utils.py          # 工具函数
├── requirements.txt      # 依赖列表
└── tsne.py               # 额外的t-SNE可视化脚本
```

## 模型架构

- **生成器(Generator)**：基于双向LSTM网络，根据随机噪声生成时序数据
- **判别器(Discriminator)**：采用1D卷积网络，区分真实数据与生成数据
- **训练策略**：标准GAN对抗训练，交替优化生成器和判别器

## 评估指标

系统提供多种量化评估指标，评估结果保存在`outputs/<数据集名称>/visualization/evaluation_results.txt`：

- **Jensen-Shannon散度**：衡量真实数据与生成数据分布的相似度
- **预测评分**：评估生成数据的时序预测能力
- **判别指标分数**：衡量生成数据的逼真度（越低越好）
- **特征方差差异**：比较真实与生成数据的特征方差

## 可视化结果

可视化结果保存在`outputs/<数据集名称>/visualization/`目录：

- `pca_tsne.png`：包含PCA和t-SNE两种降维可视化，对比真实数据（蓝色）与生成数据（红色）的分布

## 扩展与定制

1. 新增数据集类型：
   - 在`data_loaders/data_loader.py`中实现新的数据集加载器
   - 继承`BaseDataLoader`并实现`_preprocess`方法

2. 调整模型参数：
   - 生成器参数在`models/traj_timegan.py`的`Generator`类中修改
   - 判别器参数在`models/traj_timegan.py`的`Discriminator`类中修改
   - 训练参数（批次大小、学习率等）在`src/train.py`的`TrajTimeGANTrainer`类中调整

3. 新增评估指标：
   - 在`src/metrics.py`的`MetricEvaluator`类中添加新的评估方法
   - 在`evaluate_all`方法中集成新指标
