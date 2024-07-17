import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, mean_squared_error
import tqdm
import sys
import os
import gc
import argparse
import warnings
warnings.filterwarnings('ignore')

# （2） 探索性数据分析（EDA）
# 在数据准备阶段，主要读取训练数据和测试数据，并进行基本的数据展示。
# 数据简单介绍：
# - 其中id为房屋id，
# - dt为日标识，训练数据dt最小为11，不同id对应序列长度不同；
# - type为房屋类型，通常而言不同类型的房屋整体消耗存在比较大的差异；
# - target为实际电力消耗，也是我们的本次比赛的预测目标。
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 进行简单的可视化分析，对数据有个简单的了解。
# - 不同type类型对应target的柱状图
import matplotlib.pyplot as plt
# 不同type类型对应target的柱状图
type_target_df = train.groupby('type')['target'].mean().reset_index()
plt.figure(figsize=(8, 4))
plt.bar(type_target_df['type'], type_target_df['target'], color=['blue', 'green'])
plt.xlabel('Type')
plt.ylabel('Average Target Value')
plt.title('Bar Chart of Target by Type')
plt.show()

# - id为00037f39cf的按dt为序列关于target的折线图
specific_id_df = train[train['id'] == '00037f39cf']
plt.figure(figsize=(10, 5))
plt.plot(specific_id_df['dt'], specific_id_df['target'], marker='o', linestyle='-')
plt.xlabel('DateTime')
plt.ylabel('Target Value')
plt.title("Line Chart of Target for ID '00037f39cf'")
plt.show()

# （3）特征工程
# 这里主要构建了 历史平移特征 和 窗口统计特征；每种特征都是有理可据的，具体说明如下：
# 历史平移特征：通过历史平移获取上个阶段的信息；如下图所示，可以将d-1时间的信息给到d时间，d时间信息给到d+1时间，这样就实现了平移一个单位的特征构建。
# 窗口统计特征：窗口统计可以构建不同的窗口大小，然后基于窗口范围进统计均值、最大值、最小值、中位数、方差的信息，可以反映最近阶段数据的变化情况。
# 可以将d时刻之前的三个时间单位的信息进行统计构建特征给我d时刻。

# 合并训练数据和测试数据，并进行排序
data = pd.concat([test, train], axis=0, ignore_index=True)
data = data.sort_values(['id', 'dt'], ascending=False).reset_index(drop=True)

# 历史平移
for i in range(10, 30):
    data[f'last{i}_target'] = data.groupby(['id'])['target'].shift(i)

# 窗口统计
data[f'win3_mean_target'] = (data['last10_target'] + data['last11_target'] + data['last12_target']) / 3

# 进行数据切分
train = data[data.target.notnull()].reset_index(drop=True)
test = data[data.target.isnull()].reset_index(drop=True)

# 确定输入特征
train_cols = [f for f in data.columns if f not in ['id', 'target']]
# （4）模型训练与测试集预测
# 这里选择使用Lightgbm模型，也是通常作为数据挖掘比赛的基线模型，在不需要过程调参的情况的也能得到比较稳定的分数。
# 另外需要注意的训练集和验证集的构建：因为数据存在时序关系，所以需要严格按照时序进行切分，
# 这里选择原始给出训练数据集中dt为30之后的数据作为训练数据，之前的数据作为验证数据，
# 这样保证了数据不存在穿越问题（不使用未来数据预测历史数据）
def time_model(lgb, train_df, test_df, cols):
    # 训练集和验证集切分
    trn_x, trn_y = train_df[train_df.dt >= 31][cols], train_df[train_df.dt >= 31]['target']
    val_x, val_y = train_df[train_df.dt <= 30][cols], train_df[train_df.dt <= 30]['target']
    # 构建模型输入数据
    train_matrix = lgb.Dataset(trn_x, label=trn_y)
    valid_matrix = lgb.Dataset(val_x, label=val_y)
    # lightgbm参数
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'min_child_weight': 5,
        'num_leaves': 2 ** 5,
        'lambda_l2': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 4,
        'learning_rate': 0.05,
        'seed': 2024,
        'nthread': 16,
        'verbose': -1,
    }
    # 训练模型
    model = lgb.train(lgb_params, train_matrix, 50000, valid_sets=[train_matrix, valid_matrix],
                      categorical_feature=[], verbose_eval=500, early_stopping_rounds=500)
    # 验证集和测试集结果预测
    val_pred = model.predict(val_x, num_iteration=model.best_iteration)
    test_pred = model.predict(test_df[cols], num_iteration=model.best_iteration)
    # 离线分数评估
    score = mean_squared_error(val_pred, val_y)
    print(score)

    return val_pred, test_pred


lgb_oof, lgb_test = time_model(lgb, train, test, train_cols)

# 保存结果文件到本地
test['target'] = lgb_test
test[['id', 'dt', 'target']].to_csv('submit.csv', index=None)