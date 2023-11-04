#!/home/rock/.conda/envs/openke/bin/python

from collections import defaultdict
import random

STATIC_OPTIMIZAER_TYPE = ["sgd","adagrad","adadelta","adam"]
STATIC_OPTIMIZER_LR = [2e-5, 1.0]
STATIC_P_NORM = [1,2]
STATIC_NORM_FLAG = [0,1,2]
STATIC_LOSS_TYPE = ["marginloss", "sigmoidloss", "softplusloss"]
STATIC_ADV_TEMPERATURE = [0,2]

POPULATION_SIZE = 100
MAX_FE = 250
ETA = 1  # 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1

class Solution:
    def __init__(self) -> None:
        self.random_init()
        self.n = 0      #排名
        self.S = []
        self.rank = 0
        self.distance = 0

        # 重载小于号“<”
    def __lt__(self, other):
        v1 = self.getObj()
        v2 = other.getObj()
        for i in range(len(v1)):
            if v1[i] > v2[i]:
                return 0  # 但凡有一个位置是 v1大于v2的 直接返回0,如果相等的话比较下一个目标值
        return 1

    def __len__(self):
        return 6

    def random_init(self):
        self.x1_optimizer_type = random.randint(0,len(STATIC_OPTIMIZAER_TYPE)-1)
        self.x2_optimizer_lr = random.uniform(STATIC_OPTIMIZER_LR[0], STATIC_OPTIMIZER_LR[1])
        self.x3_p_norm = random.randint(0,len(STATIC_P_NORM)-1)
        self.x4_norm_flag = random.randint(0,len(STATIC_NORM_FLAG)-1)
        self.x5_loss_type = random.randint(0,len(STATIC_LOSS_TYPE)-1)
        self.x6_adv_temperature = random.uniform(STATIC_ADV_TEMPERATURE[0], STATIC_ADV_TEMPERATURE[1])

    def display(self):
        print("x1:{} x2:{} x3:{} x4:{} x5:{} x6:{} distance:{} rank:{} n:{}".format(
            STATIC_OPTIMIZAER_TYPE[self.x1_optimizer_type], 
            self.x2_optimizer_lr, 
            STATIC_P_NORM[self.x3_p_norm], 
            STATIC_NORM_FLAG[self.x4_norm_flag], 
            STATIC_LOSS_TYPE[self.x5_loss_type], 
            self.x6_adv_temperature, self.distance, self.rank, self.n))
    
    def displayIndex(self):
        self.obj1 = self.obj1_func1()
        self.obj2 = self.obj2_func2()
        print("obj1:{} obj2:{} x1:{} x2:{} x3:{} x4:{} x5:{} x6:{} distance:{} rank:{} n:{}".format(
            self.obj1,
            self.obj2,
            self.x1_optimizer_type, 
            self.x2_optimizer_lr, 
            self.x3_p_norm, 
            self.x4_norm_flag, 
            self.x5_loss_type, 
            self.x6_adv_temperature, self.distance, self.rank, self.n))
        
    def getObj(self):
        self.obj1 = self.obj1_func1()
        self.obj2 = self.obj2_func2()
        return [self.obj1, self.obj2]

    def equal(self, other):
        return (self.x1_optimizer_type == other.x1_optimizer_type) & (self.x2_optimizer_lr == other.x2_optimizer_lr) & (self.x3_p_norm == other.x3_p_norm) &(self.x4_norm_flag == other.x4_norm_flag) &(self.x5_loss_type == other.x5_loss_type) &(self.x6_adv_temperature == other.x6_adv_temperature)

    def obj1_func1(self):
        return -(self.x1_optimizer_type + self.x2_optimizer_lr + self.x3_p_norm + self.x4_norm_flag + self.x5_loss_type + self.x6_adv_temperature)
    
    def obj2_func2(self):
        return -(self.x1_optimizer_type + self.x2_optimizer_lr + self.x3_p_norm + self.x4_norm_flag - self.x5_loss_type + self.x6_adv_temperature)

    def setBeta(self, beta, i, parent1, parent2):
        i = i + 1
        if i == 1:
            self.x1_optimizer_type = 0.5 * ((1 + beta) * parent1.x1_optimizer_type + (1 - beta) * parent2.x1_optimizer_type)
        elif i == 2:
            self.x2_optimizer_lr = 0.5 * ((1 + beta) * parent1.x2_optimizer_lr + (1 - beta) * parent2.x2_optimizer_lr)
        elif i == 3:
            self.x3_p_norm = 0.5 * ((1 + beta) * parent1.x3_p_norm + (1 - beta) * parent2.x3_p_norm)
        elif i == 4:
            self.x4_norm_flag = 0.5 * ((1 + beta) * parent1.x4_norm_flag + (1 - beta) * parent2.x4_norm_flag)
        elif i == 5:
            self.x5_loss_type = 0.5 * ((1 + beta) * parent1.x5_loss_type + (1 - beta) * parent2.x5_loss_type)
        elif i == 6:
            self.x6_adv_temperature = 0.5 * ((1 + beta) * parent1.x6_adv_temperature + (1 - beta) * parent2.x6_adv_temperature)
    
    def setMutiDelta(self, i, delta):
        i = i + 1
        if i == 1:
            self.x1_optimizer_type += delta
        elif i == 2:
            self.x2_optimizer_lr += delta
        elif i == 3:
            self.x3_p_norm  += delta
        elif i == 4:
            self.x4_norm_flag += delta
        elif i == 5:
            self.x5_loss_type += delta
        elif i == 6:
            self.x6_adv_temperature += delta

    def bound_process(self):
        self.x1_optimizer_type = int(self.x1_optimizer_type)
        if self.x1_optimizer_type < 0:
            self.x1_optimizer_type = 0
        elif self.x1_optimizer_type > len(STATIC_OPTIMIZAER_TYPE)-1:
            self.x1_optimizer_type = len(STATIC_OPTIMIZAER_TYPE)-1
        
        if self.x2_optimizer_lr < STATIC_OPTIMIZER_LR[0]:
            self.x2_optimizer_lr = STATIC_OPTIMIZER_LR[0]
        elif self.x2_optimizer_lr > STATIC_OPTIMIZER_LR[1]:
            self.x2_optimizer_lr = STATIC_OPTIMIZER_LR[1]

        self.x3_p_norm = int(self.x3_p_norm)
        if self.x3_p_norm < 0:
            self.x3_p_norm = 0
        elif self.x3_p_norm > len(STATIC_P_NORM)-1:
            self.x3_p_norm = len(STATIC_P_NORM)-1

        self.x4_norm_flag = int(self.x4_norm_flag)
        if self.x4_norm_flag < 0:
            self.x4_norm_flag = 0
        elif self.x4_norm_flag > len(STATIC_NORM_FLAG)-1:
            self.x4_norm_flag = len(STATIC_NORM_FLAG)-1

        self.x5_loss_type = int(self.x5_loss_type)
        if self.x5_loss_type < 0:
            self.x5_loss_type = 0
        elif self.x5_loss_type > len(STATIC_LOSS_TYPE)-1:
            self.x5_loss_type = len(STATIC_LOSS_TYPE)-1

        if self.x6_adv_temperature < STATIC_ADV_TEMPERATURE[0]:
            self.x6_adv_temperature = STATIC_ADV_TEMPERATURE[0]
        elif self.x6_adv_temperature > STATIC_ADV_TEMPERATURE[1]:
            self.x6_adv_temperature = STATIC_ADV_TEMPERATURE[1]

def fast_non_dominated_sort(P):
    """
    非支配排序
    :param P: 种群 P
    :return F: F=(F_1, F_2, ...) 将种群 P 分为了不同的层， 返回值类型是dict，键为层号，值为 List 类型，存放着该层的个体
    """
    F = defaultdict(list)

    for p in P:
        p.S = []
        p.n = 0
        for q in P:
            if p < q:  # if p dominate q
                p.S.append(q)  # Add q to the set of solutions dominated by p
            elif q < p:
                p.n += 1  # Increment the domination counter of p
        if p.n == 0:
            p.rank = 1
            F[1].append(p)

    i = 1
    while F[i]:
        Q = []
        for p in F[i]:
            for q in p.S:
                q.n = q.n - 1
                if q.n == 0:
                    q.rank = i + 1
                    Q.append(q)
        i = i + 1
        F[i] = Q

    return F

def crowding_distance_assignment(L):
    """ 传进来的参数应该是L = F(i)，类型是List"""
    l = len(L)  # number of solution in F

    for i in range(l):
        L[i].distance = 0  # initialize distance

    # m = 1
    L.sort(key=lambda x: x.obj1)  # sort using each objective value
    # for i in range(l):
    #     L[i].displayIndex()
    # print("*******************************", l)
    
    L[0].distance = float('inf')
    L[l - 1].distance = float('inf')  # so that boundary points are always selected
    # 排序是由小到大的，所以最大值和最小值分别是 L[l-1] 和 L[0]
    f_max = L[l - 1].obj1
    f_min = L[0].obj1

    for i in range(1, l - 1):  # for all other points
        L[i].distance = L[i].distance + (L[i + 1].obj1 - L[i - 1].obj1) / (f_max - f_min)

    # m = 2
    L.sort(key=lambda x: x.obj2)  # sort using each objective value
    L[0].distance = float('inf')
    L[l - 1].distance = float('inf')  # so that boundary points are always selected
    # 排序是由小到大的，所以最大值和最小值分别是 L[l-1] 和 L[0]
    f_max = L[l - 1].obj2
    f_min = L[0].obj2

    for i in range(1, l - 1):  # for all other points
        L[i].distance = L[i].distance + (L[i + 1].obj2 - L[i - 1].obj2) / (f_max - f_min)

    # for m in L[0].objective.keys():
    #     L.sort(key=lambda x: x.objective[m])  # sort using each objective value
    #     L[0].distance = float('inf')
    #     L[l - 1].distance = float('inf')  # so that boundary points are always selected

    #     # 排序是由小到大的，所以最大值和最小值分别是 L[l-1] 和 L[0]
    #     f_max = L[l - 1].objective[m]
    #     f_min = L[0].objective[m]

    #     for i in range(1, l - 1):  # for all other points
    #         L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)

        # 虽然发生概率较小，但为了防止除0错，当bug发生时请替换为以下代码
        # if f_max != f_min:
        #     for i in range(1, l - 1):  # for all other points
        #         L[i].distance = L[i].distance + (L[i + 1].objective[m] - L[i - 1].objective[m]) / (f_max - f_min)

def binary_tournament(ind1, ind2):
    """
    二元锦标赛
    :param ind1:个体1号
    :param ind2: 个体2号
    :return:返回较优的个体
    """
    if ind1.rank != ind2.rank:  # 如果两个个体有支配关系，即在两个不同的rank中，选择rank小的
        return ind1 if ind1.rank < ind2.rank else ind2
    elif ind1.distance != ind2.distance:  # 如果两个个体rank相同，比较拥挤度距离，选择拥挤读距离大的
        return ind1 if ind1.distance > ind2.distance else ind2
    else:  # 如果rank和拥挤度都相同，返回任意一个都可以
        return ind1

def make_new_pop(P, eta):
    """
    use select,crossover and mutation to create a new population Q
    :param P: 父代种群
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return Q : 子代种群
    """
    popnum = len(P)
    Q = []
    # binary tournament selection
    for i in range(int(popnum / 2)):
        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent1
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent1 = binary_tournament(P[i], P[j])

        # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent2
        i = random.randint(0, popnum - 1)
        j = random.randint(0, popnum - 1)
        parent2 = binary_tournament(P[i], P[j])

        while (parent1.equal(parent2) ):  # 如果选择到的两个父代完全一样，则重选另一个
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = binary_tournament(P[i], P[j])

        # parent1 和 parent1 进行交叉，变异 产生 2 个子代
        Two_offspring = crossover_mutation(parent1, parent2, eta)

        # 产生的子代进入子代种群
        Q.append(Two_offspring[0])
        Q.append(Two_offspring[1])
    return Q


def crossover_mutation(parent1, parent2, eta):
    """
    交叉方式使用二进制交叉算子（SBX），变异方式采用多项式变异（PM）
    :param parent1: 父代1
    :param parent2: 父代2
    :param eta: 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
    :param bound_min: 定义域下限
    :param bound_max: 定义域上限
    :param objective_fun: 目标函数
    :return: 2 个子代
    """
    poplength = len(parent1)

    offspring1 = Solution()
    offspring2 = Solution()

    # 二进制交叉
    for i in range(poplength):
        rand = random.random()
        beta = (rand * 2) ** (1 / (eta + 1)) if rand < 0.5 else (1 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
        offspring1.setBeta(beta, i, parent1, parent2)
        offspring2.setBeta(beta, i, parent1, parent2)
        # offspring1.solution[i] = 0.5 * ((1 + beta) * parent1.solution[i] + (1 - beta) * parent2.solution[i])
        # offspring2.solution[i] = 0.5 * ((1 - beta) * parent1.solution[i] + (1 + beta) * parent2.solution[i])

    # 多项式变异
    # TODO 变异的时候只变异一个，不要两个都变，不然要么出现早熟现象，要么收敛速度巨慢 why？
    for i in range(poplength):
        mu = random.random()
        delta = (2 * mu) ** (1 / (eta + 1)) if mu < 0.5 else (1 - (2 * (1 - mu)) ** (1 / (eta + 1)))
        offspring1.setMutiDelta(i, delta)
        # offspring1.solution[i] = offspring1.solution[i] + delta

    # 定义域越界处理
    offspring1.bound_process()
    offspring2.bound_process()
    # offspring1.bound_process(bound_min, bound_max)
    # offspring2.bound_process(bound_min, bound_max)

    # 计算目标函数值
    offspring1.getObj()
    offspring2.getObj()
    # offspring1.calculate_objective(objective_fun)
    # offspring2.calculate_objective(objective_fun)

    return [offspring1, offspring2]

def main():
    Population = []

    for i in range(POPULATION_SIZE):
        Population.append(Solution())
    
    # 否 -> 非支配排序
    fast_non_dominated_sort(Population)
    Q = make_new_pop(Population, ETA)

    P_t = Population  # 当前这一届的父代种群
    Q_t = Q  # 当前这一届的子代种群

    for gen_cur in range(MAX_FE):
        R_t = P_t + Q_t  # combine parent and offspring population
        F = fast_non_dominated_sort(R_t)

        P_n = []  # 即为P_t+1,表示下一届的父代
        i = 1
        while len(P_n) + len(F[i]) < POPULATION_SIZE:  # until the parent population is filled
            crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
            P_n = P_n + F[i]  # include ith non dominated front in the parent pop
            i = i + 1  # check the next front for inclusion
        F[i].sort(key=lambda x: x.distance)  # sort in descending order using <n，因为本身就在同一层，所以相当于直接比拥挤距离
        P_n = P_n + F[i][:POPULATION_SIZE - len(P_n)]
        Q_n = make_new_pop(P_n, ETA)  # use selection,crossover and mutation to create a new population Q_n

        # 求得下一届的父代和子代成为当前届的父代和子代，，进入下一次迭代 《=》 t = t + 1
        P_t = P_n
        Q_t = Q_n

    for i in range(POPULATION_SIZE):
        print("index {} -- ".format(i), end="")
        # Population[i].display()
        Population[i].displayIndex()

main()



