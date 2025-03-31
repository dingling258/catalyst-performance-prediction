import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
import warnings
import joblib

import matplotlib.pyplot as plt
import matplotlib as mpl
import platform

# 设置中文字体
if platform.system() == 'Windows':
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 优先使用的中文字体
elif platform.system() == 'Darwin':
    # macOS系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti SC']
else:
    # Linux系统
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']

# 通用设置，适用于所有系统
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
mpl.rcParams['font.family'] = 'sans-serif'

warnings.filterwarnings('ignore')

# 设置工作目录
os.chdir('D:/chemistry')

# 首先查看Excel文件的所有表头
print("查看Excel文件标题行...")
df_headers = pd.read_excel('data.xlsx', nrows=3)
print(df_headers)

# 根据第一行的情况确定正确的标题行
print("\n尝试跳过前两行读取数据...")
df = pd.read_excel('data.xlsx', header=None, skiprows=2)

# 查看数据
print("\n读取到的数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())

# 给列设置有意义的名称
# 推断列名
col_names = []
for i in range(df.shape[1]):
    if i == 0:
        col_names.append('样本ID')
    elif i == 1:
        col_names.append('Pt(wt%)')
    elif i == 2:
        col_names.append('Fe(wt%)')
    elif i == 3:
        col_names.append('Sn(wt%)')
    elif i == 4:
        col_names.append('Zn(wt%)')
    elif i == 5:
        col_names.append('Ga(wt%)')
    elif i == 6:
        col_names.append('In(wt%)')
    elif i == 7:
        col_names.append('Cu(wt%)')
    elif i == 8:
        col_names.append('Ca(wt%)')
    elif i == 9:
        col_names.append('Co(wt%)')
    elif i == 10:
        col_names.append('Mn(wt%)')
    elif i == 11:
        col_names.append('Ni(wt%)')
    elif i == 12:
        col_names.append('Ce(wt%)')
    elif i == 13:
        col_names.append('Ge(wt%)')
    elif i == 14:
        col_names.append('K(wt%)')
    elif i == 15:
        col_names.append('Bi(wt%)')
    elif i == 16:
        col_names.append('La(wt%)')
    elif i == 17:
        col_names.append('Y(wt%)')
    elif i == 18:
        col_names.append('T(℃)')
    elif i == 19:
        col_names.append('预处理时长(h)')
    elif i == 20:
        col_names.append('丙烷流速(mL/min)')
    elif i == 21:
        col_names.append('催化剂质量(g)')
    elif i == 22:
        col_names.append('WHSV(h-1)')
    elif i == 23:
        col_names.append('丙烷体积分数(%)')
    elif i == 24:
        col_names.append('氢气体积分数(%)')
    elif i == 25:
        col_names.append('载体')
    elif i == 26:
        col_names.append('负载S/限域C')
    elif i == 27:
        col_names.append('理论收率(%)')
    elif i == 28:
        col_names.append('初始丙烷转化率(%)')
    elif i == 29:
        col_names.append('初始丙烯选择性(%)')
    elif i == 30:
        col_names.append('初始丙烯收率(%)')
    elif i == 31:
        col_names.append('实际收率与理论收率比值(%)')
    elif i == 32:
        col_names.append('丙烷转化率1h(%)')
    elif i == 33:
        col_names.append('丙烷转化比1h(%)')
    elif i == 34:
        col_names.append('丙烷转化率10h(%)')
    elif i == 35:
        col_names.append('丙烷转化比10h(%)')
    elif i == 36:
        col_names.append('丙烷转化率50h(%)')
    elif i == 37:
        col_names.append('丙烷转化比50h(%)')
    elif i == 38:
        col_names.append('STY(催)(h-1)')
    elif i == 39:
        col_names.append('STY(Pt)(h-1)')
    else:
        col_names.append(f'Column_{i}')

# 设置列名
df.columns = col_names
print("\n设置的列名:", col_names)

# 处理公式列
for col in df.columns:
    # 检查是否包含公式
    has_formula = False
    try:
        has_formula = df[col].astype(str).str.contains('=').any()
    except:
        continue

    if has_formula:
        print(f"列 {col} 包含公式，正在处理...")
        df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and str(x).startswith('=') else x)

# 定义自变量列
X_cols = ['Pt(wt%)', 'Fe(wt%)', 'Sn(wt%)', 'Zn(wt%)', 'Ga(wt%)', 'In(wt%)',
          'Cu(wt%)', 'Ca(wt%)', 'Co(wt%)', 'Mn(wt%)', 'Ni(wt%)', 'Ce(wt%)',
          'Ge(wt%)', 'K(wt%)', 'Bi(wt%)', 'La(wt%)', 'Y(wt%)', 'T(℃)',
          '预处理时长(h)', '丙烷流速(mL/min)', '催化剂质量(g)', 'WHSV(h-1)',
          '丙烷体积分数(%)', '氢气体积分数(%)', '载体', '负载S/限域C']

# 定义因变量列（按重要性排序）
y_cols_primary = ['实际收率与理论收率比值(%)']  # 主要因变量
y_cols_secondary = ['丙烷转化率1h(%)', '丙烷转化比1h(%)']  # 次要因变量
y_cols_other = ['理论收率(%)', '初始丙烷转化率(%)', '初始丙烯选择性(%)', '初始丙烯收率(%)',
                '丙烷转化率10h(%)', '丙烷转化比10h(%)', '丙烷转化率50h(%)', '丙烷转化比50h(%)',
                'STY(催)(h-1)', 'STY(Pt)(h-1)']  # 其他因变量

# 合并所有因变量
all_y_cols = y_cols_primary + y_cols_secondary + y_cols_other

# 检查自变量列是否存在
missing_x_cols = [col for col in X_cols if col not in df.columns]
if missing_x_cols:
    print(f"警告：以下自变量列不存在: {missing_x_cols}")
    # 移除不存在的列
    X_cols = [col for col in X_cols if col in df.columns]

# 检查因变量列是否存在
existing_y_primary = [col for col in y_cols_primary if col in df.columns]
existing_y_secondary = [col for col in y_cols_secondary if col in df.columns]
existing_y_other = [col for col in y_cols_other if col in df.columns]
existing_y_cols = existing_y_primary + existing_y_secondary + existing_y_other

if not existing_y_primary:
    print(f"警告：主要因变量 {y_cols_primary} 不存在!")
    if existing_y_secondary:
        print(f"使用次要因变量 {existing_y_secondary} 作为主要因变量")
        existing_y_primary = existing_y_secondary[:1]  # 使用第一个次要因变量作为主要因变量
    elif existing_y_other:
        print(f"使用其他因变量 {existing_y_other[0]} 作为主要因变量")
        existing_y_primary = [existing_y_other[0]]  # 使用第一个其他因变量作为主要因变量
    else:
        raise ValueError("无法找到任何因变量!")

print(f"\n自变量列 ({len(X_cols)}):")
print(X_cols)
print(f"\n因变量列总计 ({len(existing_y_cols)}):")
print(f"主要因变量: {existing_y_primary}")
print(f"次要因变量: {existing_y_secondary}")
print(f"其他因变量: {existing_y_other}")

# 选择数据
X_raw = df[X_cols].copy()
y_raw_all = df[existing_y_cols].copy()
y_raw_primary = df[existing_y_primary].copy()  # 只选择主要因变量

# 检查数据
print(f"\nX_raw 形状: {X_raw.shape}")
print(f"y_raw_all 形状: {y_raw_all.shape}")
print(f"y_raw_primary 形状: {y_raw_primary.shape}")
print("\nX_raw 前5行:")
print(X_raw.head())
print("\ny_raw_primary 前5行:")
print(y_raw_primary.head())
print("\ny_raw_all 前5行:")
print(y_raw_all.head())

# 明确识别分类特征
categorical_columns = ['载体', '负载S/限域C']

# 确保分类特征保持为字符串类型
for col in categorical_columns:
    if col in X_raw.columns:
        X_raw[col] = X_raw[col].astype(str)
        print(f"已将 {col} 设置为分类特征")

# 转换数据类型 (尝试将所有能转成数值的列转为数值)
for col in X_raw.columns:
    if col not in categorical_columns:
        try:
            X_raw[col] = pd.to_numeric(X_raw[col], errors='coerce')
            print(f"列 {col} 转换为数值类型")
        except:
            print(f"列 {col} 无法转换为数值类型")

for col in y_raw_all.columns:
    try:
        y_raw_all[col] = pd.to_numeric(y_raw_all[col], errors='coerce')
        print(f"列 {col} 转换为数值类型")
    except:
        print(f"列 {col} 无法转换为数值类型")

# 处理'/'值
for col in X_raw.columns:
    if X_raw[col].dtype == 'object' and col not in categorical_columns:
        X_raw[col] = X_raw[col].replace('/', np.nan)

for col in y_raw_all.columns:
    if y_raw_all[col].dtype == 'object':
        y_raw_all[col] = y_raw_all[col].replace('/', np.nan)

# 区分数值特征和分类特征
numeric_features = [col for col in X_raw.columns if col not in categorical_columns]
categorical_features = [col for col in X_raw.columns if col in categorical_columns]

print(f"\n数值特征 ({len(numeric_features)}):\n", numeric_features)
print(f"\n分类特征 ({len(categorical_features)}):\n", categorical_features)

# 查看分类特征的唯一值
for cat_feat in categorical_features:
    unique_values = X_raw[cat_feat].unique()
    print(f"\n{cat_feat} 的唯一值 ({len(unique_values)}):\n", unique_values)

# 创建预处理管道
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # 使用中位数填充数值型缺失值
    ('scaler', StandardScaler())  # 标准化
])

# 创建分类特征处理流程
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # 使用最频繁值填充分类型缺失值
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # 独热编码
])

# 创建列转换器，明确指定哪些列是数值特征，哪些是分类特征
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=True  # 启用详细特征名称输出
)

# 应用预处理
X_processed = preprocessor.fit_transform(X_raw)

# 获取转换后的特征名称
transformed_feature_names = preprocessor.get_feature_names_out()
print(f"\n转换后的特征名称: {transformed_feature_names}")
print(f"转换后的特征数量: {len(transformed_feature_names)}")

# 保存预处理器及特征名称映射
joblib.dump(preprocessor, 'preprocessor.pkl')
# 特别保存转换后的特征名称，以便后续使用
np.save('transformed_feature_names.npy', transformed_feature_names)

print(f"\n处理后的特征维度: {X_processed.shape}")
print(f"处理后的特征数量: {len(transformed_feature_names)}")

# 检查处理后是否有NaN值
nan_count = np.isnan(X_processed).sum()
if nan_count > 0:
    print(f"警告：处理后的特征中仍有 {nan_count} 个NaN值")

# 将因变量中的NaN值替换为中位数
y_imputer_all = SimpleImputer(strategy='median')
y_processed_all = y_imputer_all.fit_transform(y_raw_all)

y_imputer_primary = SimpleImputer(strategy='median')
y_processed_primary = y_imputer_primary.fit_transform(y_raw_primary)

print(f"\ny_processed_all 形状: {y_processed_all.shape}")
print(f"y_processed_primary 形状: {y_processed_primary.shape}")

# 划分训练集和测试集 (分别为所有因变量和主要因变量创建数据集)
X_train, X_test, y_train_all, y_test_all, y_train_primary, y_test_primary = train_test_split(
    X_processed, y_processed_all, y_processed_primary, test_size=0.2, random_state=42
)

print("\n训练集和测试集的形状:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train_all: {y_train_all.shape}")
print(f"y_test_all: {y_test_all.shape}")
print(f"y_train_primary: {y_train_primary.shape}")
print(f"y_test_primary: {y_test_primary.shape}")

# 对比处理前后的数据分布 (最多显示6个特征)
show_features = min(6, len(numeric_features))
if show_features > 0:
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_features[:show_features]):
        plt.subplot(2, 3, i + 1)

        # 处理前的分布
        sns.histplot(X_raw[col].dropna(), kde=True, color='blue', alpha=0.5, label='原始数据')

        # 处理后的分布（需要反向标准化才能在原始尺度上比较）
        try:
            idx = numeric_features.index(col)
            standardized_values = X_processed[:, idx]
            plt.axvline(np.median(X_raw[col].dropna()), color='red', linestyle='--', label='中位数')
        except:
            print(f"无法绘制 {col} 的处理后分布")

        plt.title(f'{col} 分布')
        plt.legend()

    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300)
    plt.close()

# 可视化主要因变量分布
plt.figure(figsize=(15, 5))
for i, col in enumerate(existing_y_primary):
    plt.subplot(1, len(existing_y_primary), i + 1)
    sns.histplot(y_raw_all[col].dropna(), kde=True)
    plt.title(f'{col} 分布 (主要因变量)')

plt.tight_layout()
plt.savefig('target_primary_distribution.png', dpi=300)
plt.close()

# 可视化所有因变量分布
if len(existing_y_cols) > 1:
    n_rows = (len(existing_y_cols) + 2) // 3  # 每行最多3个图
    plt.figure(figsize=(15, 5 * n_rows))
    for i, col in enumerate(existing_y_cols):
        plt.subplot(n_rows, 3, i + 1)
        sns.histplot(y_raw_all[col].dropna(), kde=True)
        title = f'{col} 分布'
        if col in existing_y_primary:
            title += ' (主要因变量)'
        elif col in existing_y_secondary:
            title += ' (次要因变量)'
        plt.title(title)

    plt.tight_layout()
    plt.savefig('target_all_distribution.png', dpi=300)
    plt.close()

# 检查特征之间的相关性
if len(numeric_features) > 1:
    # 选择相关性最强的特征
    max_features = min(15, len(numeric_features))
    corr_matrix = X_raw[numeric_features].corr().abs()

    # 获取相关系数的平均值，按降序排列
    mean_corr = corr_matrix.mean().sort_values(ascending=False)
    top_features = mean_corr.index[:max_features].tolist()

    # 绘制热图
    plt.figure(figsize=(12, 10))
    top_corr = corr_matrix.loc[top_features, top_features]
    sns.heatmap(top_corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('前15个特征相关性矩阵')
    plt.tight_layout()
    plt.savefig('feature_correlation.png', dpi=300)
    plt.close()

# 保存处理后的数据
np.save('X_processed.npy', X_processed)
np.save('y_processed_all.npy', y_processed_all)
np.save('y_processed_primary.npy', y_processed_primary)
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train_all.npy', y_train_all)
np.save('y_test_all.npy', y_test_all)
np.save('y_train_primary.npy', y_train_primary)
np.save('y_test_primary.npy', y_test_primary)
np.save('target_names_all.npy', np.array(existing_y_cols, dtype=object))
np.save('target_names_primary.npy', np.array(existing_y_primary, dtype=object))

# 创建一个详细的特征映射，以便后续分析
feature_mapping = {
    'original_numeric_features': numeric_features,
    'original_categorical_features': categorical_features,
    'transformed_feature_names': transformed_feature_names.tolist(),
    'categorical_values': {feat: X_raw[feat].unique().tolist() for feat in categorical_features}
}

# 保存列名映射以供后续使用
column_mapping = {
    'feature_names': transformed_feature_names.tolist(),
    'feature_mapping': feature_mapping,
    'target_names_all': existing_y_cols,
    'target_names_primary': existing_y_primary,
    'target_names_secondary': existing_y_secondary,
    'target_names_other': existing_y_other
}
joblib.dump(column_mapping, 'column_mapping.pkl')

print("\n数据预处理完成，处理后的数据已保存。")
print(f"主要因变量: {existing_y_primary}")
print(f"次要因变量: {existing_y_secondary}")
print(f"其他因变量: {existing_y_other}")
print(f"转换后的特征数量: {len(transformed_feature_names)}")
print(f"特征名称已保存，可在后续分析中使用")