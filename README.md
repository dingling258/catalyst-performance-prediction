# 催化剂性能预测模型

本项目使用机器学习方法为丙烷脱氢(PDH)反应开发高性能催化剂配方预测模型。通过分析催化剂组成、载体类型与反应条件等因素对催化性能的影响，构建多种预测模型以指导实验设计。

## 项目背景

丙烷脱氢是一种重要的工业过程，通过该过程可以将丙烷转化为更有价值的丙烯。开发高效的催化剂对提高反应效率和降低能耗至关重要。本项目旨在通过机器学习方法，分析现有催化剂性能数据，建立性能预测模型，并推荐可能具有更高性能的新催化剂配方。

## 主要特点

- 全面的数据预处理流程，包括特征工程、缺失值处理和标准化
- 多模型对比评估(随机森林、KNN、梯度提升、XGBoost、ElasticNet)
- 特征重要性分析，识别关键催化剂组分和反应条件
- 超参数优化提升模型性能
- 基于模型输出推荐新催化剂配方

## 数据集

本项目使用实验室收集的催化剂性能数据集，包含以下主要特征：
- 金属组分含量(Pt, Fe, Sn, Zn, Ga等)
- 载体类型(Al₂O₃, 分子筛等)
- 反应条件(温度, WHSV, 气体组成等)
- 性能指标(转化率, 选择性, 稳定性等)

## 主要发现

- Ga含量是影响催化性能的重要金属组分
- 催化剂质量和WHSV是最关键的操作参数
- Al₂O₃和SSZ-13分子筛是性能较好的载体
- 优化后的KNN模型达到0.37的R²预测性能

## 使用方法

1. 克隆仓库:
git clone https://github.com/your-username/catalyst-performance-prediction.git
cd catalyst-performance-prediction
2. 安装依赖:   
pip install -r requirements.txt
3. 运行预处理脚本:
python src/1.data_preprocessing.py
4. 运行模型选择与评估脚本:
python src/2.model_selection.py

## 项目结构

- `src/data_preprocessing.py`: 数据清洗与特征工程
- `src/model_selection.py`: 模型训练、评估与可视化
- `results/`: 包含所有生成的图表与结果
- `models/`: 保存训练好的模型文件

## 技术栈

- Python 3.8+
- scikit-learn
- XGBoost
- pandas
- NumPy
- Matplotlib
- Seaborn

## 许可证

本项目采用MIT许可证 - 详情请见LICENSE文件