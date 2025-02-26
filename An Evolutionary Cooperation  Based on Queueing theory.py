import math
import random
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt


# 策略类
class Strategy:

    def __init__(self, theStrategy, StateTime, UsedTime=-1):
        self.theStrategy = theStrategy
        self.StateTime = StateTime
        self.UsedTime = UsedTime

    def setUsedTime(self, inputTime):
        self.UsedTime = inputTime

    def getUsedTime(self):
        if self.UsedTime == -1:
            exit()
        return self.UsedTime

# 策略队列，循环队列实现
class StrategiesQueue:
    def __init__(self, size):
        self.size = size
        self.queue = [-1] * size
        self.front = self.rear = -1

    def enqueue(self, item):
        if (self.rear + 1) % self.size == self.front:
            pass
        elif self.front == -1:
            self.front = self.rear = 0
            self.queue[self.rear] = item
        else:
            self.rear = (self.rear + 1) % self.size
            self.queue[self.rear] = item

    def dequeue(self,NowTime):
        if self.front == -1:
            return -1
        elif self.front == self.rear:
            item = self.queue[self.front]
            item.setUsedTime(NowTime)
            self.front = self.rear = -1
            return item
        else:
            item = self.queue[self.front]
            item.setUsedTime(NowTime)
            self.front = (self.front + 1) % self.size
            return item

    def display(self):
        if self.front == -1:
            print("队列为空")
        elif self.rear >= self.front:
            for i in range(self.front, self.rear + 1):
                print(self.queue[i].theStrategy, end=" ")
            print()
        else:
            for i in range(self.front, self.size):
                print(self.queue[i].theStrategy, end=" ")
            for i in range(0, self.rear + 1):
                print(self.queue[i].theStrategy, end=" ")
            print()

    def is_empty(self):
        return self.front == -1


IndividualNum = 1000  # 个体数量
IteratedNum = 1000 # 迭代次数
AverageNum = 1  # 实验平均次数

delta = 0.01  # 选择强度

result_C = []  # 结果
result_D = []  # 结果
result_L = []  # 结果

AverageDegree = 4

# 图生成
G = nx.watts_strogatz_graph(IndividualNum, AverageDegree, 0.2)
# G = nx.watts_strogatz_graph(IndividualNum, 2, 0)
# G = nx.erdos_renyi_graph(IndividualNum, 0.4)
# G = nx.barabasi_albert_graph(IndividualNum,2)
# G = nx.random_regular_graph(d=AverageDegree, n=IndividualNum)


# 个体不同速率更新
# LambdaList = [Lambda[1] for Lambda in nx.degree(G)]
# MuList = [Mu[1] for Mu in nx.degree(G)]

# 个体相同速率更新
LambdaList = [0.4 for _ in range(IndividualNum)]
MuList = [1.4 for _ in range(IndividualNum)]


# 博弈参数
b = 1.1
R, S, T, P = 1, 0, b, 0


# 获取个体收益
def get_payoff(G, node):
    payoff = 0

    for neighbor in list(G.neighbors(node)):

        # 当邻居或自身没有策略时，收益为0
        if (NowStrategiesList[node] == -1) or (NowStrategiesList[neighbor] == -1):
            continue

        if NowStrategiesList[node].theStrategy == 1 and NowStrategiesList[neighbor].theStrategy == 1:
            payoff += R
        elif NowStrategiesList[node].theStrategy == 1 and NowStrategiesList[neighbor].theStrategy == 0:
            payoff += S
        elif NowStrategiesList[node].theStrategy == 0 and NowStrategiesList[neighbor].theStrategy == 1:
            payoff += T
        elif NowStrategiesList[node].theStrategy == 0 and NowStrategiesList[neighbor].theStrategy == 0:
            payoff += P

    return payoff


# 用于判断每个时刻策略队列中的策略是否弹出
def Update(G, NowTime):

    # StrategiesQueueList[10].display()
    for UpdateNode in G.nodes:
        if NowStrategiesList[UpdateNode] == -1:     # 如果此时个体没有策略，直接弹出就好。若策略队列为空，则弹出空，这不影响
            NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue(NowTime)
        else:       # 否则判断当前策略是否超时，超时就让新的策略顶上来
            if NowTime - NowStrategiesList[UpdateNode].StateTime >= NowStrategiesList[UpdateNode].UsedTime:
                NowStrategiesList[UpdateNode] = StrategiesQueueList[UpdateNode].dequeue(NowTime)
    # StrategiesQueueList[10].display()
    # print("________")

# 基于费米函数的模仿策略
def ImitateUpdate(G, NowTime):

    # 计算全局收益
    Payoffs = np.zeros(G.number_of_nodes())
    for node in G.nodes:
        Payoffs[node] += get_payoff(G,node)


    for UpdateNode in G.nodes:

        # 是否产生更新想法，个体在时间间隔为1的时间内更新的概率是 lambda * 1
        if random.random() <= (LambdaList[UpdateNode]):

            # 剔除无策略邻居
            SelectNeighbors = [neig for neig in nx.neighbors(G, UpdateNode) if NowStrategiesList[neig] != -1]

            # 邻居全空的人就不要更新了
            AllEmpty = 1
            for neig in SelectNeighbors:
                if not NowStrategiesList[neig] == -1:
                    AllEmpty = 0
                    break
            if AllEmpty == 1:
                continue


            # 随机挑选一个邻居
            Neighbor = random.choice(SelectNeighbors)

            # 获取两个个体收益
            NeighborPayoff = Payoffs[Neighbor]
            SelfPayoff = Payoffs[UpdateNode]

            # 设置边界条件防止指数运算溢出
            if SelfPayoff - NeighborPayoff < -5:
                StrategiesQueueList[UpdateNode].enqueue(Strategy(NowStrategiesList[Neighbor].theStrategy, random.expovariate(MuList[UpdateNode])))
                continue
            if SelfPayoff - NeighborPayoff > 5:
                StrategiesQueueList[UpdateNode].enqueue(Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode])))
                continue

            # 基于费米函数的模仿规则
            if random.random() <= 1 / (1 + math.exp((SelfPayoff - NeighborPayoff) / 0.1)):
                StrategiesQueueList[UpdateNode].enqueue(Strategy(NowStrategiesList[Neighbor].theStrategy, random.expovariate(MuList[UpdateNode])))
                continue
            else:   # 不模仿
                if NowStrategiesList[UpdateNode] == -1:
                    continue
                else:
                    StrategiesQueueList[UpdateNode].enqueue(Strategy(NowStrategiesList[UpdateNode].theStrategy, random.expovariate(MuList[UpdateNode])))
                    continue


# 初始化策略队列，为每个个体填充一个时间为1的策略
def StrategiesQueueListInit(StrategiesQueueList):
    total_queues = len(StrategiesQueueList)
    half = total_queues // 2

    # 随机选择一半的队列添加 0
    selected_indices = random.sample(range(total_queues), half)
    for index in selected_indices:
        StrategiesQueueList[index].enqueue(Strategy(0,1))

    # 向剩余的队列添加 1
    for index, queue in enumerate(StrategiesQueueList):
        if index not in selected_indices:
            queue.enqueue(Strategy(1,1))

    return StrategiesQueueList

# MAINLOOP
for i in range(AverageNum):

    print(f"{i}/{AverageNum}")      # 统计迭代次数便于调试

    # print(j)
    # 初始策略集合，随机生成一个合作者，合作策略持续1个time step
    NowStrategiesList = [-1 for _ in range(IndividualNum)]

    StrategiesQueueList = [StrategiesQueue(300) for _ in range(IndividualNum)]  # 策略队列声明，队列长度尽可能大防止上溢
    StrategiesQueueList = StrategiesQueueListInit(StrategiesQueueList)      # 初始化队列

    NowTime = 0     # 初始化时间

    for x in range(IteratedNum):
        print(f"{x}/{IteratedNum}")     # 统计迭代次数便于调试

        Update(G,NowTime)
        NowTime += 1
        ImitateUpdate(G, NowTime)

        # 每一轮统计各个策略数量
        NumOfC = 0
        NumOfD = 0
        NumOfL = 0

        for i in range(IndividualNum):
            if NowStrategiesList[i] == -1:
                NumOfL += 1
            elif NowStrategiesList[i].theStrategy == 1:
                NumOfC += 1
            elif NowStrategiesList[i].theStrategy == 0:
                NumOfD += 1


        result_C.append(NumOfC/(NumOfC + NumOfD))
        result_D.append(NumOfD/IndividualNum)
        result_L.append(NumOfL/IndividualNum)


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 24

# 绘制合作水平随时间变化的曲线
plt.plot(result_C , color = "green",label = 'C', linewidth=4, markersize=10)
# plt.plot(result_D , color = "red",label = 'D', linewidth=4, markersize=10)
# plt.plot(result_L , color = "gold",label = 'O', linewidth=4, markersize=10)

plt.xlabel(r'$t$')
plt.ylabel(r'$f_c$')
plt.legend()
plt.show()
