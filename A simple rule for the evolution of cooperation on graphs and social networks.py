import random
import networkx as nx

IndividualNum = 100  # 个体数量
IteratedNum = 1000  # 迭代次数
AverageNum = 10

delta = 0.01  # 选择强度

result_C = []  # 合作者入侵结果
result_D = []  # 背叛者入侵结果
result_Ave = []  # 最终的平均结果

# 图生成
# G = nx.watts_strogatz_graph(IndividualNum, 4, 0.2)
G = nx.erdos_renyi_graph(IndividualNum, 0.06)

# 博弈参数
b = 2 * G.number_of_edges() / G.number_of_nodes() + 0.1
c = 1
R, S, T, P = b - c, -c, b, 0


def get_fitness(G, node):
    payoff = 0
    for neighbor in list(G.neighbors(node)):

        if isDead[neighbor] == 1:
            continue

        if StrategiesList[node] == 1 and StrategiesList[neighbor] == 1:
            payoff += R
        elif StrategiesList[node] == 1 and StrategiesList[neighbor] == 0:
            payoff += S
        elif StrategiesList[node] == 0 and StrategiesList[neighbor] == 1:
            payoff += T
        else:
            payoff += P

    return 1 - delta + (delta * payoff)


# 随机死亡
def Death(G):
    DeathNode = random.randint(0, IndividualNum - 1)
    isDead[DeathNode] = 1

    return G, DeathNode


# Birth
def Birth(G, DeadNode):
    C_fitness = 1e-8
    D_fitness = 1e-8
    for neighbor in list(G.neighbors(DeadNode)):
        fitness = get_fitness(G, neighbor)
        if StrategiesList[neighbor] == 1:
            C_fitness += fitness
        elif StrategiesList[neighbor] == 0:
            D_fitness += fitness
    isDead[DeadNode] = 0
    if random.random() < C_fitness / (C_fitness + D_fitness):
        StrategiesList[DeadNode] = 1
    else:
        StrategiesList[DeadNode] = 0


# MAINLOOP
for i in range(AverageNum):
    print(f"{i}/{AverageNum}")
    # 合作入侵
    for _ in range(IteratedNum):
        # 初始策略集合，随机生成一个合作者
        StrategiesList = [0 for _ in range(IndividualNum)]
        StrategiesList[random.randint(0, IndividualNum - 1)] = 1

        # 死亡记录
        isDead = [0 for _ in range(IndividualNum)]

        while (sum(StrategiesList) / len(StrategiesList)) != 1 and (sum(StrategiesList) / len(StrategiesList)) != 0:
            # Death
            G, DeathNode = Death(G)
            # Birth
            Birth(G, DeathNode)
            # print(sum(StrategiesList) / len(StrategiesList))
        result_C.append(sum(StrategiesList) / len(StrategiesList))
    # 背叛入侵
    for _ in range(IteratedNum):
        # 初始策略集合，随机生成一个背叛者
        StrategiesList = [1 for _ in range(IndividualNum)]
        StrategiesList[random.randint(0, IndividualNum - 1)] = 0

        # 死亡记录
        isDead = [0 for _ in range(IndividualNum)]

        while (sum(StrategiesList) / len(StrategiesList)) != 1 and (sum(StrategiesList) / len(StrategiesList)) != 0:
            # Death
            G, DeathNode = Death(G)
            # Birth
            Birth(G, DeathNode)
            # print(sum(StrategiesList) / len(StrategiesList))
        result_D.append(sum(StrategiesList) / len(StrategiesList))
    result_Ave.append((sum(result_C) / len(result_C)) / (1 - (sum(result_D) / len(result_D))))

print("===============================================")
print(f"Pc/Pd = {sum(result_Ave) / len(result_Ave)}")
print(f"b/c = {b / c} : k={2 * G.number_of_edges() / G.number_of_nodes()}")
