import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

#节点数量

N = 1000

# 定义速率矩阵 Q
T_lambda0 = 0.02
T_lambda1 = 0.06
T_mu1 = 0.04
T_mu2 = 0.08

# 迭代步数
T = 1000000

Q = np.array([
    [-T_lambda0, T_lambda0, 0],
    [T_mu1, -(T_mu1 + T_lambda1), T_lambda1],
    [0, T_mu2, -T_mu2]
])

# 定义初始持续时间 (可以假设从状态0开始)
initial_state_time = np.array([0.0, 0.0, 0.0])

# 初始状态
current_state = 0

#记录绘图点
plot_point_S0 = []
plot_point_S1 = []
plot_point_S2 = []

# 计算每个时间点的概率分布

current_t = 0
while current_t < T:

    print("当前时间:",current_t)
    print("当前状态:", current_state)

    TransferRate = -Q[current_state, current_state]
    LifeTime = np.random.exponential(1 / TransferRate)  # 生成指数分布的状态存在时间

    current_t += LifeTime
    initial_state_time[current_state] += LifeTime

    TransferProbability = Q[current_state, :].copy()
    TransferProbability[current_state] = 0
    TransferProbability = TransferProbability / TransferRate

    if current_state == 0:
        plot_point_S0.append((current_t,initial_state_time[0]*N/current_t))
        plot_point_S1.append((current_t, initial_state_time[1]*N / current_t))
        plot_point_S2.append((current_t, initial_state_time[2]*N / current_t))
    if current_state == 1:
        plot_point_S0.append((current_t,initial_state_time[0]*N/current_t))
        plot_point_S1.append((current_t, initial_state_time[1]*N / current_t))
        plot_point_S2.append((current_t, initial_state_time[2] *N/ current_t))
    if current_state == 2:
        plot_point_S0.append((current_t,initial_state_time[0]*N/current_t))
        plot_point_S1.append((current_t, initial_state_time[1]*N / current_t))
        plot_point_S2.append((current_t, initial_state_time[2] *N/ current_t))


    current_state = random.choices([0, 1, 2], weights=TransferProbability, k=1)[0]


print(plot_point_S0[-1])
print(plot_point_S1[-1])
print(plot_point_S2[-1])
# 将二元组分解为 x 和 y 坐标
x1, y1 = zip(*plot_point_S0)
x2, y2 = zip(*plot_point_S1)
x3, y3 = zip(*plot_point_S2)

# 创建绘图
plt.figure(figsize=(10, 6))

# 设置对数坐标
plt.xscale('log')

# 绘制曲线
plt.plot(x1, y1, marker='o', label='PDG', color='blue')
plt.plot(x2, y2, marker='s', label='SGD', color='black')
plt.plot(x3, y3, marker='^', label='SHG', color='red')

# 设置标题和标签
plt.title('Curve Plot of Three Groups')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 添加图例
plt.legend()

# 显示网格
plt.grid()

# 显示图形
plt.show()