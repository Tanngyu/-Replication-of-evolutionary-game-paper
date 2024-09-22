import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

# 网络参数
num_nodes = 1000  # 节点数量
m = 2  # 每个新节点连接到的现有节点数

# 博弈参数
num_iterations = 1000  # 博弈迭代次数


b = 1.5
R = 1.0
S = 0.0
T = b
P = 0.0



# 使用Networkx创建无标度网络G
G = nx.barabasi_albert_graph(num_nodes, m)

# 策略设置 (0表示叛变，1表示合作)

strategies = np.random.choice([0, 1], size=num_nodes)   # 随机策略


# 计算每个节点的收益
def calculate_payoff(G, strategies):
    payoff = np.zeros(G.number_of_nodes())

    # 计算节点收益
    for node in G.nodes():
        # 单个节点收益
        for neighbor in G.neighbors(node):  # 挑出节点的邻居集
            if strategies[node] == 1 and strategies[neighbor] == 1:
                payoff[node] += R
            elif strategies[node] == 1 and strategies[neighbor] == 0:
                payoff[node] += S
            elif strategies[node] == 0 and strategies[neighbor] == 1:
                payoff[node] += T
            else:
                payoff[node] += P

    return payoff


# 仿真过程
cooperation_levels = []     # 统计合作水平便于绘图

for _ in range(num_iterations):
    payoff = calculate_payoff(G, strategies)
    new_strategies = strategies.copy()

    # 每个节点与邻居比较收益，并模仿收益更高的邻居的策略
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if neighbors:   # 如果有邻居
            neighbor = np.random.choice(neighbors)   # 随机挑选一个邻居来对比相互的收益
            if payoff[neighbor] > payoff[node]:     # 直接对比收益而不是通过费米法则
                new_strategies[node] = strategies[neighbor]

    strategies = new_strategies

    # 记录当前的合作水平
    cooperation_level = np.mean(strategies)  # 直接取平均值就是合作水平
    cooperation_levels.append(cooperation_level)

# 绘制合作水平随时间变化的曲线
plt.plot(cooperation_levels)
plt.xlabel('Iterations')
plt.ylabel('Cooperation Level')
plt.title('Emergence of Cooperation in a Scale-Free Network')
plt.show()
