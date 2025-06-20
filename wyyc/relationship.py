import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot  as plt 
from datetime import datetime 
 
# 数据加载与预处理 
df = pd.read_csv('data/all_data_water_tp_new.csv') 
 
# 数据清洗（根据实际情况调整）
print("缺失值统计：")
print(df.isnull().sum()) 
df = df.dropna()   # 删除含缺失值记录 
df = df.select_dtypes(include=[np.number])   # 仅保留数值型数据 
 
# 相关性计算（增加三种方法）
corr_matrix = df.corr(method='pearson')   # 默认皮尔逊相关系数 
spearman_corr = df.corr(method='spearman')   # 非参数相关 
kendall_corr = df.corr(method='kendall')    # 序数数据相关 
 
# 可视化设置 
plt.rcParams.update({ 
    'font.sans-serif':  'SimHei',  # 中文显示 
    'axes.unicode_minus':  False,   # 负号显示 
    'figure.dpi':  120,            # 输出分辨率 
    'savefig.dpi':  300            # 保存分辨率 
})
 
# 热力图增强版 
plt.figure(figsize=(12,  8))
mask = np.triu(np.ones_like(corr_matrix,  dtype=bool))  # 隐藏上三角 
heatmap = sns.heatmap(corr_matrix, 
                      mask=mask,
                      annot=True,
                      annot_kws={'size': 8},
                      fmt=".2f",
                      cmap='coolwarm',
                      center=0,
                      linewidths=0.5,
                      cbar_kws={'shrink': 0.8})
 
plt.title(f' 水质参数相关性分析（TP为重点）\n生成时间：{datetime.now().strftime("%Y-%m-%d  %H:%M")}',
          fontsize=14, pad=20)
plt.xticks(rotation=45,  ha='right')
plt.tight_layout() 
plt.savefig('img/correlation_analysis.png',  bbox_inches='tight')
plt.show() 
 
# 重点参数分析（假设目标列为TP）
tp_corr = corr_matrix['TP'].sort_values(ascending=False)
print("\nTP相关性排序：")
print(tp_corr[tp_corr.index  != 'TP'])  # 排除自相关 
 
# 进阶可视化（选重点参数绘制回归图）
significant_vars = tp_corr[abs(tp_corr) > 0.5].index.tolist() 
if 'TP' in significant_vars: 
    significant_vars.remove('TP') 
 
sns.pairplot(df[significant_vars  + ['TP']],
             kind='reg',
             plot_kws={'line_kws':{'color':'red'}, 
                      'scatter_kws': {'alpha':0.3}})
plt.suptitle(' 关键参数与TP的回归关系', y=1.02)
plt.savefig('tp_regression_analysis.png') 