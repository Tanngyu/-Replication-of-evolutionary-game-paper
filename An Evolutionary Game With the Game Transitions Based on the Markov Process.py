import math
import random
import time

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 网络参数

num_nodes = 1000     # 节点数
ws_k = 4
ws_p = 0.4


# 博弈参数
num_iterations = 1000   # 博弈迭代次数

b = 1.6
r = 0.5


#马尔科夫链参数

T_lambda0 = 0.03
T_lambda1 = 0.03
T_mu1 = 0.05
T_mu2 = 0.04

Q = np.array([
    [-T_lambda0, T_lambda0, 0],
    [T_mu1, -(T_mu1 + T_lambda1), T_lambda1],
    [0, T_mu2, -T_mu2]
])

#声誉参数

kappa = 0.1     # 非理性选择参数
delta = 0.04

# 网络创建
# G = nx.watts_strogatz_graph(num_nodes,ws_k,ws_p)
G = nx.barabasi_albert_graph(num_nodes,5)

# 初始策略设置(0表示叛变，1表示合作),随机：
strategies = np.random.choice([0, 1], size=num_nodes)

# 初始博弈分布
GameState = [[0] * num_iterations for _ in range(num_nodes)]


# 初始声誉设置
mean = 2    # 均值
std = math.sqrt(0.6)    # 标准差
reputations = np.random.normal(loc=mean, scale=std, size=num_nodes)
reputations = np.clip(reputations, 0, 4)

# 计算每个节点的博弈状态
def Game_State(GameState):
    for node in range(0, num_nodes):
        current_t = 0.0
        current_state = 0
        while current_t < num_iterations:
            (GameState[node])[int(current_t)] = current_state

            TransferRate = -Q[current_state, current_state]
            LifeTime = np.random.exponential(1 / TransferRate)  # 生成指数分布的状态存在时间

            for i in range(int(current_t),int(current_t + LifeTime)):
                if i == num_iterations:
                    break
                (GameState[node])[i] = current_state

            current_t += LifeTime

            TransferProbability = Q[current_state, :].copy()
            TransferProbability[current_state] = 0
            TransferProbability = TransferProbability / TransferRate

            current_state = random.choices([0, 1, 2], weights=TransferProbability, k=1)[0]

    return GameState

# 计算每个节点的收益
def calculate_payoff(strategies , iteration):

    payoff = np.zeros(num_nodes)  # 创建一个大小为参数的数组并初始化为0

    for node in G.nodes():
        if (GameState[node])[iteration] == 0:
            R = 1
            S = 0
            T = b
            P = 0
        elif (GameState[node])[iteration] == 1:
            R = 1
            S = 1-r
            T = 1+r
            P = 0
        else:
            R = 1
            S = -r
            T = r
            P = 0

        neighbors = list(G.neighbors(node))  # 获取当前节点的邻居节点
        for neighbor in neighbors:
            if strategies[node] == 1 and strategies[neighbor] == 1:
                payoff[node] += R
            elif strategies[node] == 1 and strategies[neighbor] == 0:
                payoff[node] += S
            elif strategies[node] == 0 and strategies[neighbor] == 1:
                payoff[node] += T
            else:
                payoff[node] += P
    return payoff


# 计算每个节点的声誉
def calculate_reputation(strategies, reputations):
    new_reputations = reputations.copy()
    for node in G.nodes():
        if strategies[node] == 1:
            new_reputations[node] += delta

        elif strategies[node] == 0:
            new_reputations[node] -= delta

        # 确保声誉在范围内
        new_reputations[node] = max(delta/2, min(new_reputations[node], 4))

    return new_reputations


# 仿真过程
cooperation_levels = []  # 用来统计合作水平，画图
GameState = Game_State(GameState)   # 获取各个节点的博弈状态

for iteration in range(num_iterations):
    print(iteration, "/", num_iterations)
    payoff = calculate_payoff(strategies,iteration)
    new_strategies = strategies.copy()

    for node in G.nodes():

        neighbors = list(G.neighbors(node))

        # 获取所有邻居的声誉
        neighbor_reputations = np.array([reputations[neighbor] for neighbor in neighbors])

        # 将邻居的声誉归一化为概率
        if np.sum(neighbor_reputations) > 0:
            probabilities = neighbor_reputations / np.sum(neighbor_reputations)
        else:
            # 如果声誉全为0，则每个邻居的概率相等
            probabilities = np.ones(len(neighbor_reputations)) / len(neighbor_reputations)

        # 根据归一化的声誉作为概率随机选择一个邻居
        chosen_neighbor = np.random.choice(neighbors, p=probabilities)

        # 获取当前节点和被选中邻居的收益
        current_payoff = payoff[node]
        neighbor_payoff = payoff[chosen_neighbor]

        # 计算选择对方策略的概率
        probability_to_switch = 1 / (1 + np.exp((current_payoff - neighbor_payoff) / kappa))

        # 以计算出的概率选择是否使用被选中邻居的策略
        if np.random.rand() < probability_to_switch:
            new_strategies[node] = strategies[chosen_neighbor]


    # 更新声誉
    reputations = calculate_reputation(strategies, reputations)
    # 更新策略
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

