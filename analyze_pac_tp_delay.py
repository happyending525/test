import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
DATA_PATH = 'data/all_data_water_tp_new.csv'
DELAY_HOURS = 10  # 可调整的延迟小时数
TP_COL = 'Effluent_TP'
TIME_COL = '_time'

# 读取数据
df = pd.read_csv(DATA_PATH)

# 过滤掉PAC1=0或PAC2=0的行
df = df[(df['PAC1'] != 0) & (df['PAC2'] != 0)]

# 确保时间列为datetime类型
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df = df.sort_values(TIME_COL).reset_index(drop=True)

# 计算延迟步数（假设数据为每小时一条）
delay_steps = DELAY_HOURS

# 分别构造PAC1和PAC2的延迟列
df['PAC1_delay'] = df['PAC1'].shift(delay_steps)
df['PAC2_delay'] = df['PAC2'].shift(delay_steps)

# 合并为一列（长数据格式）
pac1_df = df[['PAC1_delay', TP_COL]].rename(columns={'PAC1_delay': 'PAC_delay'})
pac2_df = df[['PAC2_delay', TP_COL]].rename(columns={'PAC2_delay': 'PAC_delay'})
long_df = pd.concat([pac1_df, pac2_df], axis=0, ignore_index=True)
# 去除缺失值
valid_df = long_df.dropna(subset=['PAC_delay', TP_COL])
# 过滤掉加药量小于等于50的数据
valid_df = valid_df[valid_df['PAC_delay'] > 50]

# 相关性分析
corr = valid_df['PAC_delay'].corr(valid_df[TP_COL])
print(f'延迟{DELAY_HOURS}小时后，PAC1/PAC2合并与出水总磷的相关系数: {corr:.4f}')

# 高阶多项式回归（三阶）
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
X = valid_df[['PAC_delay']].values
y = valid_df[TP_COL].values
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
reg = LinearRegression().fit(X_poly, y)
coefs = reg.coef_
intercept = reg.intercept_
print(f'三阶回归: 出水总磷 = {coefs[3]:.6f}*x^3 + {coefs[2]:.6f}*x^2 + {coefs[1]:.6f}*x + {intercept:.6f}')

# 计算最佳加药量（假设目标TP为0.275，解三次方程）
TARGET_TP = 0.275
A = coefs[3]
B = coefs[2]
C = coefs[1]
D = intercept - TARGET_TP
best_pac = None
try:
    roots = np.roots([A, B, C, D])
    candidates = [r.real for r in roots if np.isreal(r) and r.real > 0]
    if candidates:
        best_pac = min(candidates, key=lambda r: abs(r - X.mean()))
        print(f'为达到目标出水总磷={TARGET_TP}，建议加药量（延迟{DELAY_HOURS}小时后起效）为: {best_pac:.2f}')
    else:
        print('三次方程无正实数解，无法给出最佳加药量建议。')
except Exception as e:
    print('三次方程求解异常:', e)

# 可视化
plt.figure(figsize=(8,6))
plt.scatter(valid_df['PAC_delay'], valid_df[TP_COL], alpha=0.4, label='数据点', color='purple')
x_range = np.linspace(valid_df['PAC_delay'].min(), valid_df['PAC_delay'].max(), 200).reshape(-1, 1)
y_pred = reg.predict(poly.transform(x_range))
plt.plot(x_range, y_pred, color='red', label='三阶回归曲线')
plt.xlabel(f'PAC1/PAC2加药量（延迟{DELAY_HOURS}小时）')
plt.ylabel('出水总磷（mg/L）')
plt.title(f'PAC1/PAC2加药量与出水总磷关系（三阶多项式，延迟{DELAY_HOURS}小时）')
plt.legend()
plt.tight_layout()
plt.savefig('img/pac_merged_tp_delay_poly3.png')
plt.show() 