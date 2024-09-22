import random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 网络参数
GridLength = 50  # 格子网络边长
num_nodes = GridLength ** 2     # 节点数（别改）


# 博弈参数
num_iterations = 3000  # 博弈迭代次数

r = 0.7


R = 1
S = 1 - r
T = 1 + r
P = 0.0


# 绘图参数
cut = 1  # 每隔几代绘制切片

# 网络创建
G = nx.grid_2d_graph(GridLength, GridLength)

# 策略设置(0表示叛变，1表示合作)，有两种方法，通过注释来选择

# 背叛入侵
strategies = np.ones(num_nodes)
strategies[int(GridLength / 2 * GridLength + GridLength / 2)] = 0

# 随机
# strategies = np.random.choice([0, 1], size=num_nodes)
# print(strategies)


# 计算每个节点的收益
def calculate_payoff(strategies):
    payoff = np.zeros(num_nodes)  # 创建一个大小为参数的数组并初始化为0
    for node in G.nodes():
        neighbors = get_periodic_neighbors(node)  # 获取周期性邻居
        idx_node = node[0] * GridLength + node[1]

        for neighbor in neighbors:
            idx_neighbor = neighbor[0] * GridLength + neighbor[1]
            if strategies[idx_node] == 1 and strategies[idx_neighbor] == 1:
                payoff[idx_node] += R
            elif strategies[idx_node] == 1 and strategies[idx_neighbor] == 0:
                payoff[idx_node] += S
            elif strategies[idx_node] == 0 and strategies[idx_neighbor] == 1:
                payoff[idx_node] += T
            else:
                payoff[idx_node] += P

    return payoff


# 周期性邻居（NetworkX不会获取周期邻居(￣_￣|||) ）
def get_periodic_neighbors(node):
    x, y = node
    neighbors = [
        ((x + dx) % GridLength, y) for dx in [-1, 1]
    ] + [
        (x, (y + dy) % GridLength) for dy in [-1, 1]
    ]
    return neighbors


# 仿真过程
cooperation_levels = []  # 用来统计合作水平，方便画图

for iteration in range(num_iterations):

    new_strategies = strategies.copy()
    payoff = calculate_payoff(strategies)

    # 每个节点与邻居比较收益，并模仿收益更高的邻居的策略
    for node in G.nodes():
        neighbor = random.choice(get_periodic_neighbors(node))
        idx_node = node[0] * GridLength + node[1]
        max_payoff = 0
        best_strategy = strategies[idx_node]


        idx_neighbor = neighbor[0] * GridLength + neighbor[1]
        if payoff[idx_neighbor] > max_payoff:
            max_payoff = payoff[idx_neighbor]
            best_strategy = strategies[idx_neighbor]

        if max_payoff > payoff[idx_node]:
            new_strategies[idx_node] = best_strategy

    # 显示热力图（放在这里显示不了第一代，我实在想不到咋改）
    if iteration % cut == 0:
        plt.figure(figsize=(6, 6))
        data_strategies = new_strategies.reshape((GridLength ,GridLength ))
        plt.imshow(data_strategies, cmap="gray", interpolation='nearest')
        plt.title(f't = {iteration+1}')
        plt.colorbar().remove()
        plt.show()


    strategies = new_strategies


    # 记录当前的合作水平
    cooperation_level = np.mean(strategies)  # 直接取平均值就是合作水平
    cooperation_levels.append(cooperation_level)

# 绘制合作水平随时间变化的曲线
plt.plot(cooperation_levels)
plt.xlabel('t')
plt.ylabel('f_c')
plt.title('f_c')
plt.show()
