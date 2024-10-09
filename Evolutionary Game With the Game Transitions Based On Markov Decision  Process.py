import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

epsilon = 1e-3  # 一个用于收敛的小值

num_nodes = 1000
num_iterations = 200

# 博弈参数
b = 1.6
r = 0.5

# 状态集
Game_State = [0, 1, 2]  # 博弈状态集合(0:PDG,1:SDG,2:SHG)
Action = [1, 0]  # 动作集合 （1为合作，0为背叛）

# 转移概率

T_lambda0 = 0.03
T_lambda1 = 0.03
T_mu1 = 0.05
T_mu2 = 0.04

delta = 0.01


Q0 = np.array([
    [0.969962,1-0.969962, 0],
    [0.050105, 0.919807, 1 - (0.050105 + 0.919807)],
    [0, 1-0.939907, 0.939907]
])  # 合作转移率

Q1 = np.array([
    [0.969962 - delta, 1-0.969962 + delta, 0],
    [0.050105 + delta , 0.919807, 1 - (0.050105 + 0.919807) - delta],
    [0, 1-0.939907 + delta, 0.939907 - delta]
])  # 背叛转移率

gamma = 0.8  # 折扣因子

G = nx.watts_strogatz_graph(num_nodes, k=4, p=0.4)  # 初始化网络

# 初始策略设置(0表示叛变，1表示合作),随机：
Actions = np.random.choice([0, 1], size=num_nodes)
policy = [np.random.choice([0, 1], size=int(1 / (1 - gamma))) for _ in range(num_nodes)]

# 初始博弈分布
Current_GameState = [0] * num_nodes


# print("博弈分布",Current_GameState)
# print("行动分布",Actions)
# print("策略分布",policy)

# 奖励函数
def reward(node, state, action):
    game_state = state

    neighbors = G.neighbors(node)

    if game_state == 0:  # PDG
        R, S, T, P = 1, 0, b, 0
    elif game_state == 1:  # SDG
        R, S, T, P = 1, 1 - r, 1 + r, 0
    else:  # SHG
        R, S, T, P = 1, -r, r, 0

    reward = 0

    for neighbor in neighbors:

        if action == 1 and Actions[neighbor] == 1:
            reward += R
        elif action and Actions[neighbor] == 0:
            reward += S
        elif action and Actions[neighbor] == 1:
            reward += T
        else:
            reward += P

    return reward


def benefit(policy, node):
    delta = 1
    V = 0
    current_game = Current_GameState[node]
    while (delta >= 0):
        i = 0
        V += delta * reward(node, current_game, policy[i])  # 节点在t=i时，处于ti时的博弈，使用ti策略所得到的收益
        if policy[i] == 0:
            probabilities = Q0[current_game, :].copy()
            current_game = np.random.choice(Game_State, p=probabilities)
        else:
            probabilities = Q1[current_game, :].copy()
            current_game = np.random.choice(Game_State, p=probabilities)

        delta -= (1 - gamma)
        i += 1

    return V


def get_best_policy(policy_in, node):
    payoff = 0
    n = len(policy_in)

    may_policies = []
    # 总共2^n个组合
    for i in range(2 ** n):
        # 将i转换为二进制，并填充0到n位
        binary_array = [(i >> j) & 1 for j in range(n)]
        may_policies.append(binary_array[::-1])  # 反转以保持顺序

    best_benefit = 0
    best_policy = policy_in
    for may_best_policy in may_policies:
        if benefit(may_best_policy, node) >= best_benefit:
            best_benefit = benefit(may_best_policy, node)
            best_policy = may_best_policy

    return best_policy


def update_GameState(Old_GameState, Current_Actions):
    new_GameState = Old_GameState.copy()
    for i in range(num_nodes):
        if Current_Actions[i] == 0:
            probabilities = Q0[new_GameState[i], :].copy()
            new_GameState[i] = np.random.choice(Game_State, p=probabilities)
        else:
            probabilities = Q1[new_GameState[i], :].copy()
            new_GameState[i] = np.random.choice(Game_State, p=probabilities)

    return new_GameState


all_game_states = []  # 用于保存所有状态
F_c = []  # 用于记录合作水平
F_c.append(np.mean(Actions))

for x in range(num_iterations):

    # 获取最优策略集合
    node_best_policies = []
    for node in G.nodes:
        node_best_policy = get_best_policy(policy[node], node)
        node_best_policies.append(node_best_policy)
    # print(node_best_policies)
    node_best_policies_array = np.array(node_best_policies)
    # print("**",node_best_policies_array)
    policy = node_best_policies_array  # 更新策略集

    print("当前时刻动作", x, "/", num_iterations, "：", node_best_policies)

    # 更新博弈状态

    for time in range(int(1 / (1 - gamma))):
        all_game_states.append(Current_GameState)
        Current_Actions = []
        for node_policies_i in node_best_policies:  # 获取当前动作集用于更新策略
            Current_Actions.append(node_policies_i[time])
        Actions = Current_Actions
        Current_GameState = update_GameState(Current_GameState, Current_Actions)
    F_c.append(np.mean(Actions))
    Current_GameState = all_game_states[-1]

proportions = []

for moment in all_game_states:
    total = len(moment)
    count_0 = moment.count(0) / total
    count_1 = moment.count(1) / total
    count_2 = moment.count(2) / total
    proportions.append([count_0, count_1, count_2])

# 转换为 NumPy 数组以便处理
proportions = np.array(proportions)

# 求收敛平均
average_value = []
last_elements_PDG = proportions[-10:, 0]
average_value.append(np.mean(last_elements_PDG))
last_elements_SDG = proportions[-10:, 1]
average_value.append(np.mean(last_elements_SDG))
last_elements_SHG = proportions[-10:, 2]
average_value.append(np.mean(last_elements_SHG))

print("收敛值:", average_value)

# 绘制占比图
plt.figure(figsize=(12, 6))
plt.plot(proportions[:, 0], label='PDG', marker='o')
plt.plot(proportions[:, 1], label='SDG', marker='o')
plt.plot(proportions[:, 2], label='SHG', marker='o')

plt.title('Gi')
plt.xlabel('t')
plt.ylabel('G')
plt.legend()
plt.show()

# 绘制合作水平图
plt.plot(F_c, label='F_c', color='blue')
plt.title('F_c based on Markov Decision')
plt.xlabel('t')
plt.ylabel('F_c')
plt.legend()
plt.show()
