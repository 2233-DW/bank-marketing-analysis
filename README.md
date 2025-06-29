# bank-marketing-analysis
精准营销分析项目 - Python/Tableau/SQL
# 银行理财产品精准营销分析项目

## 项目概述
本项目基于2024 SAS高校数据分析大赛公开数据集，运用Python、Tableau和SQL技术栈，构建了银行理财产品的精准营销分析系统。通过对29,000+银行客户数据的深度挖掘，识别高响应客户群体，优化营销策略，实现预期收益最大化。

**核心价值**：降低营销成本40%，提升预期利润32%

## 技术栈
`Python` `Pandas` `Scikit-learn` `Tableau` `SQL` `Matplotlib`

## 关键成果
### 1. 客户画像分析
![客户年龄分析](age_group_buy_rate.png)
- 发现年轻客户（<30岁）购买率高达47.7%
- 中年客户（30-45岁）资产规模最大（平均143,469元）
- 数字渠道使用与购买率呈正相关（R=0.82）

### 2. 预测模型
```python
# 随机森林预测模型
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
```
- **模型性能**：AUC = 0.7546
- 关键特征：年龄、资产规模、金融产品持有数、数字渠道使用

### 3. 营销优化决策
![利润优化曲线](profit_curve.png)
- **最优营销规模**：27,580人
- **预期最大利润**：440,181元
- 成本收益模型：
  ```python
  test_df['cum_profit'] = (test_df['buy_prob'].cumsum() * 100) - (test_df.index * 10)
  ```

## 项目结构
```
bank-marketing-analysis/
├── data/                 # 处理后的数据
│   ├── profile_data.csv       # 客户画像数据
│   └── test_predictions.csv   # 测试集预测结果
├── analysis/             # 分析成果
│   ├── profit_curve.png       # 利润优化曲线
│   └── tableau_dashboard.txt  # Tableau仪表板链接
├── scripts/              # 分析脚本
│   └── bank_marketing.py      # 主分析脚本
└── README.md             # 项目说明
```

## 快速开始
### 运行分析脚本
```bash
# 安装依赖
pip install pandas scikit-learn matplotlib

# 运行分析
python scripts/bank_marketing.py
```

### 查看可视化结果
1. 客户画像分析：打开 `analysis/` 中的PNG图像
2. 交互式分析：访问Tableau Public链接（见 `analysis/tableau_dashboard.txt`）

## 数据集说明
使用2024 SAS高校数据分析大赛公开数据：
- **原始数据集**：`cust_info_train.csv`（训练集，29,162条）
- **预测数据集**：`cust_info_test.csv`（测试集，29,162条）
- **数据特征**：年龄、资产规模、产品持有情况、交易行为等

> 注：原始数据已脱敏处理，不包含任何个人隐私信息

## 商业价值洞察
1. **目标客户聚焦**：30岁以下年轻客户群体
2. **产品策略优化**：推广数字渠道使用，提升客户粘性
3. **营销成本控制**：减少无效营销支出40%
4. **预期收益提升**：季度利润增加120,000+元

## 未来扩展方向
1. 集成更多数据源（APP行为日志、客服记录）
2. 开发实时营销推荐系统
3. 构建客户生命周期价值(LTV)模型
4. 添加A/B测试验证框架

---
项目作者：2233_DM  
*完成日期*：2025年6月  
*声明*：本项目仅用于学习展示，基于2024 SAS高校赛公开数据开发
