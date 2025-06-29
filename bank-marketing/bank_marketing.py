# 全流程集成脚本
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
mpl.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 检查文件路径是否存在
def check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    return True

# 1. 数据清洗函数
def clean_data(df):
    df = df.copy()
    if 'age' in df.columns:
        df['age'] = df['age'].clip(18, 70)
    if 'aum' in df.columns:
        df['aum'] = df['aum'].fillna(df['aum'].median())
    
    # 处理资产异常值
    asset_cols = ['hold_dpst', 'hold_ploan', 'hold_invst', 'hold_fund', 'hold_ccard']
    if all(col in df.columns for col in asset_cols):
        df['financial_products'] = df[asset_cols].sum(axis=1)
    
    return df

# 2. 特征工程函数
def add_features(df):
    # 创建数字渠道使用特征
    channel_cols = ['sign_tpdep', 'sign_mbapp', 'sign_wchat', 'sign_alipay']
    if all(col in df.columns for col in channel_cols):
        df['digital_channels'] = df[channel_cols].sum(axis=1)
    
    # 创建交易活跃度特征
    if 'acct_age' in df.columns and 'dpst_tx_cnt' in df.columns and 'ccard_tx_cnt' in df.columns:
        df['transaction_activity'] = (df['dpst_tx_cnt'] + df['ccard_tx_cnt']) / (df['acct_age'] + 1)
    
    return df

# 3. 加载训练集
try:
    train_path = "D:\\M_QSL_data\\cust_info_train.csv"
    check_path(train_path)
    train_df = pd.read_csv(train_path, encoding='GB18030')
    print(f"训练集加载成功, 样本数: {len(train_df)}")
    
    train_df = clean_data(train_df)
    train_df = add_features(train_df)
    print("训练集清洗和特征工程完成")
except Exception as e:
    print(f"训练集处理出错: {str(e)}")
    exit()

# 4. 客户画像分析
try:
    # 创建年龄分组
    train_df['age_group'] = pd.cut(
        train_df['age'],
        bins=[0, 30, 45, 60, 100],
        labels=['<30', '30-45', '45-60', '60+']
    )
    
    # 创建客户画像数据
    profile_data = train_df.groupby('age_group', observed=True).agg(
        avg_aum=('aum', 'mean'),
        avg_products=('financial_products', 'mean'),
        buy_rate=('buy', 'mean'),
        customer_count=('cust_id', 'count')
    ).reset_index()
    
    # 添加金融产品分组
    train_df['product_group'] = pd.cut(
        train_df['financial_products'],
        bins=[0, 1, 2, 5, 10],
        labels=['低(0-1)', '中(2)', '高(3-5)', '极高(6+)']
    )
    
    # 保存画像数据供Tableau使用
    profile_data.to_csv("profile_data.csv", index=False)
    print("客户画像数据已保存: profile_data.csv")
    print(profile_data)
except Exception as e:
    print(f"客户画像生成出错: {str(e)}")

# 5. 建模预测
# 模型评估
from sklearn.metrics import roc_auc_score
try:
    # 选择特征
    features = []
    if 'age' in train_df.columns: features.append('age')
    if 'aum' in train_df.columns: features.append('aum')
    if 'financial_products' in train_df.columns: features.append('financial_products')
    if 'digital_channels' in train_df.columns: features.append('digital_channels')
    
    if not features:
        raise ValueError("没有可用的特征列")
    
    X = train_df[features].fillna(0)
    y = train_df['buy']
    
    # 简化建模（避免复杂调参）
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=42
    )
    model.fit(X, y)
    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    print(f"模型AUC: {auc:.4f}")
    # 添加预测概率到训练集
    train_df['buy_prob'] = model.predict_proba(X)[:, 1]
    print("模型训练完成")
except Exception as e:
    print(f"建模过程出错: {str(e)}")

# 6. 测试集预测
try:
    test_path = "D:\M_QSL_data\cust_info_train.csv"
    check_path(test_path)
    test_df = pd.read_csv(test_path, encoding='GB18030')
    print(f"测试集加载成功, 样本数: {len(test_df)}")
    
    test_df = clean_data(test_df)
    test_df = add_features(test_df)
    
    # 确保特征一致
    for feature in features:
        if feature not in test_df.columns:
            test_df[feature] = 0
    
    # 预测测试集
    test_df['buy_prob'] = model.predict_proba(test_df[features].fillna(0))[:, 1]
    test_df[['cust_id', 'buy_prob']].to_csv("test_predictions.csv", index=False)
    print("测试集预测完成: test_predictions.csv")
except Exception as e:
    print(f"测试集处理出错: {str(e)}")

# 7. 成本收益分析
try:
    # 对测试集按概率排序
    test_df = test_df.sort_values('buy_prob', ascending=False)
    test_df = test_df.reset_index(drop=True)
    
    # 计算累计利润
    test_df['cum_customers'] = test_df.index + 1
    test_df['cum_expected_response'] = test_df['buy_prob'].cumsum()
    test_df['cum_profit'] = (test_df['cum_expected_response'] * 100) - (test_df['cum_customers'] * 10)
    
    # 找到最优营销点
    optimal_idx = test_df['cum_profit'].idxmax()
    optimal_customers = test_df.loc[optimal_idx, 'cum_customers']
    max_profit = test_df.loc[optimal_idx, 'cum_profit']
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.plot(test_df['cum_customers'], test_df['cum_profit'])
    plt.axvline(x=optimal_customers, color='r', linestyle='--', label=f'最优点: {optimal_customers}人')
    plt.title("营销利润优化曲线")
    plt.xlabel("营销客户数")
    plt.ylabel("预期利润(元)")
    plt.legend()
    plt.grid(True)
    plt.savefig("profit_curve.png")
    print(f"成本收益分析完成: 最优营销人数 = {optimal_customers}, 最大利润 = {max_profit:,.0f}元")
    print(f"图表已保存: profit_curve.png")
except Exception as e:
    print(f"成本收益分析出错: {str(e)}")

train_df[['cust_id', 'age_group', 'digital_channels', 'buy']].to_csv("digital_channels_data.csv", index=False)

print("\n全流程完成:")
print("1. 客户画像数据: profile_data.csv")
print("2. 测试集预测结果: test_predictions.csv")
print("数字渠道数据已保存: digital_channels_data.csv")
print("3. 利润优化曲线: profit_curve.png")
print("4. 建议: 整合")

