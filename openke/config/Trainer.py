# coding:utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import os
import time
import sys
import datetime
import ctypes
import json
import numpy as np
import copy
from tqdm import tqdm
import random

from openke.ea.nsgaii import NSGAII
from openke.ea.solution import Solution

POPULATION_SIZE = 10
MAX_FE = 50
ETA = 1  # 变异分布参数，该值越大则产生的后代个体逼近父代的概率越大。Deb建议设为 1
EA_LEN = 10

class Trainer(object):

    def __init__(self, 
                 model = None,
                 data_loader = None,
                 train_times = 1000,
                 alpha = 0.5,
                 use_gpu = True,
                 opt_method = "sgd",
                 save_steps = None,
                 checkpoint_dir = None):

        self.work_threads = 8
        self.train_times = train_times

        self.opt_method = opt_method
        self.optimizer = None
        self.lr_decay = 0
        self.weight_decay = 0
        self.alpha = alpha

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.save_steps = save_steps
        self.checkpoint_dir = checkpoint_dir

    def train_one_step(self, data):
        self.optimizer.zero_grad()
        loss = self.model({
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
            'batch_y': self.to_var(data['batch_y'], self.use_gpu),
            'mode': data['mode']
        })
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def run(self):
        if self.use_gpu:
            self.model.cuda()

        print(time.asctime(time.localtime(time.time())))
        # 首先有一个Model: init_model
        # 计算init_Model的正样本得分p0， 负样本得分q0
        # 初始化种群
        Population = []
        ea_alg = NSGAII()
        ea_alg.set_init_model(self.model)

        all_data = []
        # index = 0
        # for i in range(10):
        #     data = copy.deepcopy(self.data_loader.getNextData())
        #     all_data.append(data)

        init_solution = Solution(self.model, same = True, opt_type = self.opt_method, lr = self.alpha, data_loader = self.data_loader, data = all_data)
        Population.append(init_solution)
        for i in range(POPULATION_SIZE-1):
            Population.append(Solution(self.model, same = False, data_loader = self.data_loader, data = all_data))

        # 否 -> 非支配排序
        ea_alg.fast_non_dominated_sort(Population)
        Q = ea_alg.make_new_pop(Population, ETA)
        P_t = Population  # 当前这一届的父代种群
        Q_t = Q  # 当前这一届的子代种群

        for gen_cur in range(MAX_FE):
            print("**********FE {}****************".format(gen_cur))
            for i in range(POPULATION_SIZE):
                print("Solution %d" % i)
                P_t[i].display()
                print("-----")
            print("----------------------------------------")
            R_t = P_t + Q_t  # combine parent and offspring population
            F = ea_alg.fast_non_dominated_sort(R_t)

            P_n = []  # 即为P_t+1,表示下一届的父代
            i = 1
            while len(P_n) + len(F[i]) < POPULATION_SIZE:  # until the parent population is filled
                ea_alg.crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
                P_n = P_n + F[i]  # include ith non dominated front in the parent pop
                i = i + 1  # check the next front for inclusion
            F[i].sort(key=lambda x: x.distance)  # sort in descending order using <n，因为本身就在同一层，所以相当于直接比拥挤距离
            P_n = P_n + F[i][:POPULATION_SIZE - len(P_n)]
            Q_n = ea_alg.make_new_pop(P_n, ETA)  # use selection,crossover and mutation to create a new population Q_n

                # 求得下一届的父代和子代成为当前届的父代和子代，，进入下一次迭代 《=》 t = t + 1
            P_t = P_n
            Q_t = Q_n

        print("**********************")
        index = 0
        min_obj_flag = P_t[index].obj2
        for i in range(POPULATION_SIZE):
            P_t[i].displayObj()
            if P_t[i].obj2 < min_obj_flag:
                min_obj_flag = P_t[i].obj2
                index = i

        # index = random.randint(0, POPULATION_SIZE - 1)
        print("select index {}".format(index))
        P_t[index].display()
        self.model = P_t[index].model
        self.optimizer = P_t[index].optimizer
        print("Finish initializing...")
        print(time.asctime(time.localtime(time.time())))

        training_range = tqdm(range(self.train_times))
        for epoch in training_range:
            res = 0.0
            for data in self.data_loader:
                loss = self.train_one_step(data)
                res += loss
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))
            
            if self.save_steps and self.checkpoint_dir and (epoch + 1) % self.save_steps == 0:
                print("Epoch %d has finished, saving..." % (epoch))
                self.model.save_checkpoint(os.path.join(self.checkpoint_dir + "-" + str(epoch) + ".ckpt"))

            # if epoch > 500 and (epoch % 100 == 0):
            #     Population = P_t
            #     data = self.data_loader.getCurData()
            #     for i in range(POPULATION_SIZE):
            #         Population[i].train_one_step(data)
                
            #     # 否 -> 非支配排序
            #     ea_alg.fast_non_dominated_sort(Population)
            #     Q = ea_alg.make_new_pop(Population, ETA)
            #     P_t = Population  # 当前这一届的父代种群
            #     Q_t = Q  # 当前这一届的子代种群

            #     for gen_cur in range(MAX_FE):
            #         # print("**********FE {}****************".format(gen_cur))
            #         R_t = P_t + Q_t  # combine parent and offspring population
            #         F = ea_alg.fast_non_dominated_sort(R_t)

            #         P_n = []  # 即为P_t+1,表示下一届的父代
            #         i = 1
            #         while len(P_n) + len(F[i]) < POPULATION_SIZE:  # until the parent population is filled
            #             ea_alg.crowding_distance_assignment(F[i])  # calculate crowding-distance in F_i
            #             P_n = P_n + F[i]  # include ith non dominated front in the parent pop
            #             i = i + 1  # check the next front for inclusion
            #         F[i].sort(key=lambda x: x.distance)  # sort in descending order using <n，因为本身就在同一层，所以相当于直接比拥挤距离
            #         P_n = P_n + F[i][:POPULATION_SIZE - len(P_n)]
            #         Q_n = ea_alg.make_new_pop(P_n, ETA)  # use selection,crossover and mutation to create a new population Q_n

            #             # 求得下一届的父代和子代成为当前届的父代和子代，，进入下一次迭代 《=》 t = t + 1
            #         P_t = P_n
            #         Q_t = Q_n
            #     print("**********************")
            #     for i in range(POPULATION_SIZE):
            #         P_t[i].displayObj()

            #     index = random.randint(0, POPULATION_SIZE - 1)
            #     print("select index {}".format(index))
            #     P_t[index].displayObj()
            #     self.model = P_t[index].model
            #     self.optimizer = P_t[index].optimizer
            #     print("Finish select index...")




    def set_model(self, model):
        self.model = model
    
    def get_model(self):
        return self.model.model

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_lr_decay(self, lr_decay):
        self.lr_decay = lr_decay

    def set_weight_decay(self, weight_decay):
        self.weight_decay = weight_decay

    def set_opt_method(self, opt_method):
        self.opt_method = opt_method

    def set_train_times(self, train_times):
        self.train_times = train_times

    def set_save_steps(self, save_steps, checkpoint_dir = None):
        self.save_steps = save_steps
        if not self.checkpoint_dir:
            self.set_checkpoint_dir(checkpoint_dir)

    def set_checkpoint_dir(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
