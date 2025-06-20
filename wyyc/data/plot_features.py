import pandas as pd
import matplotlib.pyplot as plt

# 读取数据集
df = pd.read_csv('data/all_data_water_tp_new.csv')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 绘制每个特征的图表
for column in df.columns:
    if column != '_time':  # 排除时间列
        plt.figure(figsize=(10, 6))
        plt.plot(df['_time'], df[column], label=column)
        plt.title(f'{column} 随时间的变化')
        plt.xlabel('时间')
        plt.ylabel('值')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'img/{column}_plot.png')
        plt.close()

print('所有特征的图表已保存到 img 目录。') 