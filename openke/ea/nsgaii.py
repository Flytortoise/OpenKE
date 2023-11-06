import random
from collections import defaultdict
from openke.ea.solution import Solution

class NSGAII:
    def __init__(self) -> None:
        pass

    def fast_non_dominated_sort(self, P):
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

    def crowding_distance_assignment(self, L):
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

    def binary_tournament(self, ind1, ind2):
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

    def make_new_pop(self, P, eta):
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
            parent1 = self.binary_tournament(P[i], P[j])

            # 从种群中随机选择两个个体，进行二元锦标赛，选择出一个 parent2
            i = random.randint(0, popnum - 1)
            j = random.randint(0, popnum - 1)
            parent2 = self.binary_tournament(P[i], P[j])

            while (parent1.equal(parent2) ):  # 如果选择到的两个父代完全一样，则重选另一个
                i = random.randint(0, popnum - 1)
                j = random.randint(0, popnum - 1)
                parent2 = self.binary_tournament(P[i], P[j])

            # parent1 和 parent1 进行交叉，变异 产生 2 个子代
            Two_offspring = self.crossover_mutation(parent1, parent2, eta)

            # 产生的子代进入子代种群
            Q.append(Two_offspring[0])
            Q.append(Two_offspring[1])
        return Q


    def crossover_mutation(self, parent1, parent2, eta):
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

        offspring1 = Solution(same = False, data_loader=parent1.data_loader, data=parent1.data)
        offspring2 = Solution(same = False, data_loader=parent1.data_loader, data=parent1.data)

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
