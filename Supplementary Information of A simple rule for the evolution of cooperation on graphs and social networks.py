import random
import networkx as nx

IndividualNum = 100  # 个体数量
IteratedNum = 1000  # 迭代次数
AverageNum = 100

delta = 0.01  # 选择强度

result_C = []  # 合作者入侵结果
result_D = []  # 背叛者入侵结果
result_Ave = []  # 最终的平均结果

AverageDegree = 6

# 图生成
# G = nx.watts_strogatz_graph(IndividualNum, AverageDegree, 0.2)
G = nx.erdos_renyi_graph(IndividualNum, AverageDegree/IndividualNum)
while not (nx.is_connected(G) and 2 * G.number_of_edges() / G.number_of_nodes() == AverageDegree):
    # G = nx.watts_strogatz_graph(IndividualNum, AverageDegree, 0.2)
    G = nx.erdos_renyi_graph(IndividualNum, AverageDegree/IndividualNum)

print(f"k={2 * G.number_of_edges() / G.number_of_nodes()}")

# 博弈参数
b = 2 * G.number_of_edges() / G.number_of_nodes() + 1 + 2  # k+2+0.01 满足 b/c > k+2
c = 1
R, S, T, P = b - c, -c, b, 0


def get_fitness(G, node):
    payoff = 0
    for neighbor in list(G.neighbors(node)):

        if StrategiesList[node] == 1 and StrategiesList[neighbor] == 1:
            payoff += R
        elif StrategiesList[node] == 1 and StrategiesList[neighbor] == 0:
            payoff += S
        elif StrategiesList[node] == 0 and StrategiesList[neighbor] == 1:
            payoff += T
        else:
            payoff += P

    return 1 - delta + (delta * payoff)


# 模仿
def ImitateUpdate(G):
    UpdateNode = random.randint(0, IndividualNum - 1)
    C_fitness = 0
    D_fitness = 0
    # 加f_0
    self_fitness = get_fitness(G, UpdateNode)
    if StrategiesList[UpdateNode] == 1:
        C_fitness += self_fitness
    else:
        D_fitness += self_fitness
    # 求KaFa和KbFb
    for neighbor in G.neighbors(UpdateNode):
        neighbor_fitness = get_fitness(G, neighbor)
        if StrategiesList[neighbor] == 1:
            C_fitness += neighbor_fitness
        else:
            D_fitness += neighbor_fitness
    total = C_fitness + D_fitness
    if total == 0:
        return
    # 计算切换概率
    if StrategiesList[UpdateNode] == 1:
        if random.random() < D_fitness / total:
            StrategiesList[UpdateNode] = 0
    else:
        if random.random() < C_fitness / total:
            StrategiesList[UpdateNode] = 1


# MAINLOOP
for i in range(AverageNum):
    print(f"{i}/{AverageNum}")
    # 合作入侵
    for _ in range(IteratedNum):
        # 初始策略集合，随机生成一个合作者
        StrategiesList = [0 for _ in range(IndividualNum)]
        StrategiesList[random.randint(0, IndividualNum - 1)] = 1
        while (sum(StrategiesList) / len(StrategiesList)) != 1 and (sum(StrategiesList) / len(StrategiesList)) != 0:
            # Update
            ImitateUpdate(G)
        result_C.append(sum(StrategiesList) / len(StrategiesList))

    result_Ave.append(((sum(result_C) / len(result_C)) * IndividualNum))
    print(f"Pc*N = {sum(result_Ave) / len(result_Ave)} >? 1")
