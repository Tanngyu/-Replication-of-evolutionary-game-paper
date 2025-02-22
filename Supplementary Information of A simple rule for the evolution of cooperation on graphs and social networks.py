import random
import networkx as nx

IndividualNum = 100  # 个体数量
IteratedNum = 100  # 迭代次数
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
b = 2 * G.number_of_edges() / G.number_of_nodes() + 2 + 0.01 # k+2+0.01 满足 b/c > k+2
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

    if StrategiesList[node] == 1:
        payoff += b-c

    return 1 - delta + (delta * payoff)


# 模仿
def ImitateUpdate(G):

    UpdateNode = random.randint(0, IndividualNum - 1)
    Neighbor = random.choice(list(nx.neighbors(G,UpdateNode)))

    NeighborFitness = get_fitness(G, Neighbor)
    SelfFitness = get_fitness(G, UpdateNode)
    total = NeighborFitness + SelfFitness

    if random.random() < NeighborFitness / total:
        StrategiesList[UpdateNode] = StrategiesList[Neighbor]


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
