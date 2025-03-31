import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import platform
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
import xgboost as xgb
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

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

# 设置工作目录
os.chdir('D:/chemistry')

# 加载预处理后的数据
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train_primary = np.load('y_train_primary.npy')
y_test_primary = np.load('y_test_primary.npy')
target_names_primary = np.load('target_names_primary.npy', allow_pickle=True)

# 加载列映射信息
column_mapping = joblib.load('column_mapping.pkl')
print(f"特征数量: {len(column_mapping['feature_names'])}")
print(f"主要因变量: {column_mapping['target_names_primary']}")

print(f"X_train 形状: {X_train.shape}")
print(f"y_train_primary 形状: {y_train_primary.shape}")


# 定义评估指标函数
def evaluate_model(model, X, y_true, model_name, target_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'Model': model_name,
        'Target': target_name,
        'RMSE': rmse,
        'MAE': mae,
        'R^2': r2
    }


# 定义要使用的模型
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
}

# 使用5折交叉验证评估模型
results = []
cv = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n开始对主要因变量进行5折交叉验证评估...")
for target_idx, target_name in enumerate(target_names_primary):
    y_train_target = y_train_primary[:, target_idx].reshape(-1, 1)
    y_test_target = y_test_primary[:, target_idx].reshape(-1, 1)

    print(f"\n评估因变量: {target_name}")

    for model_name, model in models.items():
        try:
            # 使用交叉验证评估
            cv_scores = cross_val_score(model, X_train, y_train_target.ravel(),
                                        cv=cv, scoring='neg_root_mean_squared_error')
            cv_rmse = -np.mean(cv_scores)

            # 在整个训练集上训练模型
            model.fit(X_train, y_train_target.ravel())

            # 评估训练集和测试集性能
            train_metrics = evaluate_model(model, X_train, y_train_target, model_name, target_name)
            test_metrics = evaluate_model(model, X_test, y_test_target, model_name, target_name)

            # 添加CV评分
            train_metrics['CV_RMSE'] = cv_rmse
            test_metrics['CV_RMSE'] = cv_rmse

            # 添加集合类型标签
            train_metrics['Set'] = 'Train'
            test_metrics['Set'] = 'Test'

            results.append(train_metrics)
            results.append(test_metrics)

            print(f"{model_name}: CV RMSE = {cv_rmse:.4f}, Test R^2 = {test_metrics['R^2']:.4f}")

            # 保存模型
            joblib.dump(model, f'model_{model_name.replace(" ", "_").lower()}_{target_name}.pkl')
        except Exception as e:
            print(f"训练 {model_name} 模型时出错: {str(e)}")

# 转换为DataFrame
results_df = pd.DataFrame(results)
print("\n模型评估结果:")
print(results_df)

# 可视化模型性能
plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='R^2', hue='Set', data=results_df)
plt.title('不同模型的R²评分对比', fontsize=14)
plt.xticks(rotation=30, fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('model_r2_comparison.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
sns.barplot(x='Model', y='RMSE', hue='Set', data=results_df)
plt.title('不同模型的RMSE评分对比', fontsize=14)
plt.xticks(rotation=30, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('model_rmse_comparison.png', dpi=300)
plt.close()

# 找出测试集上表现最好的模型
best_test_models = results_df[results_df['Set'] == 'Test'].sort_values('R^2', ascending=False)
best_model_name = best_test_models['Model'].iloc[0]
print(f"\n在测试集上表现最好的模型是: {best_model_name}, R^2 = {best_test_models['R^2'].iloc[0]:.4f}")

# 获取特征重要性（针对支持特征重要性的模型）
feature_importance_models = ['Random Forest', 'Gradient Boosting', 'XGBoost']

for target_idx, target_name in enumerate(target_names_primary):
    print(f"\n分析目标变量: {target_name} 的特征重要性")

    # 加载特征名称
    try:
        transformed_feature_names = column_mapping['feature_names']
        print(f"转换后的特征总数: {len(transformed_feature_names)}")
    except Exception as e:
        print(f"加载特征名称时出错: {str(e)}")
        transformed_feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

    # 创建一个DataFrame来存储所有模型的特征重要性
    all_importance_df = pd.DataFrame({'Feature': transformed_feature_names})

    # 分析每个模型的特征重要性
    for model_name in feature_importance_models:
        try:
            model_file = f'model_{model_name.replace(" ", "_").lower()}_{target_name}.pkl'
            if os.path.exists(model_file):
                model = joblib.load(model_file)

                # 获取特征重要性
                if hasattr(model, 'feature_importances_'):
                    feature_importance = model.feature_importances_

                    # 检查特征重要性长度是否与特征数匹配
                    if len(feature_importance) == len(transformed_feature_names):
                        # 添加到DataFrame
                        all_importance_df[model_name] = feature_importance

                        # 单独可视化每个模型的特征重要性
                        importance_df = pd.DataFrame({
                            'Feature': transformed_feature_names,
                            'Importance': feature_importance
                        })
                        importance_df = importance_df.sort_values('Importance', ascending=False)

                        # 仅显示前15个特征
                        plt.figure(figsize=(12, 10))
                        top_df = importance_df.head(15)
                        sns.barplot(x='Importance', y='Feature', data=top_df)
                        plt.title(f'{model_name} 模型的特征重要性 (目标变量: {target_name})', fontsize=14)
                        plt.xlabel('重要性', fontsize=12)
                        plt.ylabel('特征', fontsize=12)
                        plt.xticks(fontsize=10)
                        plt.yticks(fontsize=10)
                        plt.tight_layout()
                        plt.savefig(f'feature_importance_{model_name.lower()}_{target_name}.png', dpi=300)
                        plt.close()

                        print(f"{model_name} 模型的前10个重要特征:")
                        print(importance_df.head(10))
                    else:
                        print(
                            f"警告: {model_name} 特征重要性长度({len(feature_importance)})与特征名称数量({len(transformed_feature_names)})不匹配")
        except Exception as e:
            print(f"获取 {model_name} 模型特征重要性时出错: {str(e)}")

    # 计算平均特征重要性（如果至少有一个模型提供了特征重要性）
    importance_cols = [col for col in all_importance_df.columns if col != 'Feature']
    if importance_cols:
        # 计算平均值和标准差
        all_importance_df['Mean_Importance'] = all_importance_df[importance_cols].mean(axis=1)
        all_importance_df['Std_Importance'] = all_importance_df[importance_cols].std(axis=1)

        # 按平均重要性排序
        all_importance_df = all_importance_df.sort_values('Mean_Importance', ascending=False)

        # 可视化平均特征重要性
        plt.figure(figsize=(12, 10))
        top_features = all_importance_df.head(15)
        sns.barplot(x='Mean_Importance', y='Feature', data=top_features)
        plt.title(f'平均特征重要性 (目标变量: {target_name})', fontsize=14)
        plt.xlabel('平均重要性', fontsize=12)
        plt.ylabel('特征', fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f'feature_importance_average_{target_name}.png', dpi=300)
        plt.close()

        # 保存特征重要性
        all_importance_df.to_csv(f'feature_importance_{target_name}.csv', index=False)

        print(f"\n所有模型的平均特征重要性前15位:")
        print(all_importance_df[['Feature', 'Mean_Importance']].head(15))

# 最后保存结果摘要
results_df.to_csv('model_evaluation_results.csv', index=False)

# 创建模型性能总结
model_summary = results_df[results_df['Set'] == 'Test'].pivot_table(
    index='Model',
    values=['R^2', 'RMSE', 'MAE', 'CV_RMSE'],
    aggfunc='mean'
).sort_values('R^2', ascending=False)

print("\n模型性能总结 (测试集):")
print(model_summary)

# 保存模型性能总结
model_summary.to_csv('model_performance_summary.csv')

print("\n模型训练和评估完成!")

# 绘制预测值与真实值对比图
for target_idx, target_name in enumerate(target_names_primary):
    best_model_file = f'model_{best_model_name.replace(" ", "_").lower()}_{target_name}.pkl'
    if os.path.exists(best_model_file):
        best_model = joblib.load(best_model_file)

        # 获取预测值
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # 创建预测值与真实值对比图
        plt.figure(figsize=(12, 10))

        # 训练集
        plt.scatter(y_train_primary[:, target_idx], y_train_pred,
                    alpha=0.5, label='训练集', color='blue')

        # 测试集
        plt.scatter(y_test_primary[:, target_idx], y_test_pred,
                    alpha=0.5, label='测试集', color='red')

        # 添加对角线
        max_val = max(np.max(y_train_primary[:, target_idx]), np.max(y_test_primary[:, target_idx]))
        min_val = min(np.min(y_train_primary[:, target_idx]), np.min(y_test_primary[:, target_idx]))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')

        plt.xlabel('真实值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title(f'最佳模型 ({best_model_name}) 的预测值与真实值对比 - {target_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'prediction_vs_actual_{target_name}.png', dpi=300)
        plt.close()

# KNN模型优化
if best_model_name == 'KNN' or True:  # 无论如何都执行KNN优化
    print("\n进行KNN模型超参数优化...")
    from sklearn.model_selection import GridSearchCV

    for target_idx, target_name in enumerate(target_names_primary):
        y_train_target = y_train_primary[:, target_idx].reshape(-1, 1)
        y_test_target = y_test_primary[:, target_idx].reshape(-1, 1)

        # 定义参数网格
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        # 创建KNN模型
        knn = KNeighborsRegressor()

        # 网格搜索
        grid_search = GridSearchCV(
            knn, param_grid, cv=5,
            scoring='r2', verbose=1, n_jobs=-1
        )

        # 拟合数据
        grid_search.fit(X_train, y_train_target.ravel())

        # 打印最佳参数
        print(f"\nKNN模型针对 {target_name} 的最佳参数:")
        print(f"最佳参数: {grid_search.best_params_}")
        print(f"最佳R²分数: {grid_search.best_score_:.4f}")

        # 使用最佳参数的模型
        best_knn = grid_search.best_estimator_
        y_train_pred = best_knn.predict(X_train)
        y_test_pred = best_knn.predict(X_test)

        train_r2 = r2_score(y_train_target, y_train_pred)
        test_r2 = r2_score(y_test_target, y_test_pred)

        print(f"优化后的KNN - 训练集R²: {train_r2:.4f}")
        print(f"优化后的KNN - 测试集R²: {test_r2:.4f}")

        # 保存最佳KNN模型
        joblib.dump(best_knn, f'model_knn_optimized_{target_name}.pkl')

        # 创建优化后的KNN预测值与真实值对比图
        plt.figure(figsize=(12, 10))

        # 训练集
        plt.scatter(y_train_target, y_train_pred,
                    alpha=0.5, label='训练集', color='blue')

        # 测试集
        plt.scatter(y_test_target, y_test_pred,
                    alpha=0.5, label='测试集', color='red')

        # 添加对角线
        max_val = max(np.max(y_train_target), np.max(y_test_target))
        min_val = min(np.min(y_train_target), np.min(y_test_target))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--')

        plt.xlabel('真实值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title(f'优化后的KNN模型预测值与真实值对比 - {target_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'prediction_vs_actual_knn_optimized_{target_name}.png', dpi=300)
        plt.close()

print("\n所有模型训练、评估和优化完成!")