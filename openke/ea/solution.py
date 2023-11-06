from collections import defaultdict
import random
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
from openke.module.model import TransE
from openke.module.strategy import NegativeSampling
from openke.module.loss import MarginLoss
from openke.module.loss import SigmoidLoss
from openke.module.loss import SoftplusLoss
from openke.data import TrainDataLoader, TestDataLoader
import copy

STATIC_OPTIMIZAER_TYPE = ["sgd","adagrad","adadelta","adam"]
STATIC_OPTIMIZER_LR = [2e-5, 1.0]
STATIC_P_NORM = [1,2]
STATIC_NORM_FLAG = [0,1,2]
STATIC_LOSS_TYPE = ["marginloss", "sigmoidloss", "softplusloss"]
STATIC_ADV_TEMPERATURE = [0,2]

class Solution:
    def __init__(self, in_model = None, same = True, opt_type = 'sgd', lr = 0.1, data_loader = None, data = None) -> None:
        if same is True:
            self.x1_optimizer_type = STATIC_OPTIMIZAER_TYPE.index(opt_type)
            self.x2_optimizer_lr = lr
            self.x3_p_norm = in_model.model.getPNorm()
            self.x4_norm_flag = in_model.model.getLNorm()
            self.x5_loss_type = STATIC_LOSS_TYPE.index(in_model.loss.getType())
            self.x6_adv_temperature = in_model.loss.getAdvTemperature()
        else:
            self.random_init()

        # dataloader for training
        self.data_loader = data_loader

        if same is True:
            transe = copy.deepcopy(in_model.model)
        else:
            transe = TransE(
                ent_tot = self.data_loader.get_ent_tot(),
                rel_tot = self.data_loader.get_rel_tot(),
                dim = 200,
                p_norm = self.x3_p_norm,
                norm_flag = True)
        transe.setLNorm(self.x4_norm_flag)
        transe.setPNorm(self.x3_p_norm)
            
        # define the loss function
        self.model = NegativeSampling(
            model = transe, 
            loss = self.getLoss(),
            batch_size = self.data_loader.get_batch_size()
        )
        self.n = 0      #排名
        self.S = []
        self.rank = 0
        self.distance = 0
        self.use_gpu = True
        self.margin = nn.Parameter(torch.Tensor([5]).cuda(), requires_grad=False)

        if self.use_gpu:
            self.model.cuda()
        # 训练一次，获得obj1和obj2   TODO 每次数据是否变化
        self.data = data
        if data:
            self.train_one_step(data)
        # self.displayObj()

    # def __init__(self) -> None:
    #     self.random_init()
    #     self.n = 0      #排名
    #     self.S = []
    #     self.rank = 0
    #     self.distance = 0

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
    
    def train_one_step(self, data):
        self.optimizer = self.getOpt()
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
        self.obj2 = loss.item()
        return loss.item()
    
    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.from_numpy(x).cuda())
        else:
            return Variable(torch.from_numpy(x))
    def getLoss(self):
        loss_type = STATIC_LOSS_TYPE[self.x5_loss_type]
        if loss_type == 'marginloss':
            loss = MarginLoss(margin = 5.0)
        elif loss_type == 'sigmoidloss':
            loss = SigmoidLoss()
        else:
            loss = SoftplusLoss()
        loss.setAdvTemperature(self.x6_adv_temperature)
        return loss


    def getOpt(self):
        self.opt_method = STATIC_OPTIMIZAER_TYPE[self.x1_optimizer_type]
        self.weight_decay=0
        if self.opt_method == "Adagrad" or self.opt_method == "adagrad":
            optimizer = optim.Adagrad(
                self.model.parameters(),
                lr=self.x2_optimizer_lr,
                lr_decay=0,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adadelta" or self.opt_method == "adadelta":
            optimizer = optim.Adadelta(
                self.model.model.parameters(),
                lr=self.x2_optimizer_lr,
                weight_decay=self.weight_decay,
            )
        elif self.opt_method == "Adam" or self.opt_method == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.x2_optimizer_lr,
                weight_decay=self.weight_decay,
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr = self.x2_optimizer_lr,
                weight_decay=self.weight_decay,
            )
        return optimizer

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
    
    def displayObj(self):
        self.getObj()
        print("obj1:{}    obj2:{}".format(self.obj1, self.obj2))

    def getObj(self):
        self.obj1 = self.obj1_func1()
        self.obj2 = self.obj2_func2()
        return [self.obj1, self.obj2]

    def equal(self, other):
        return (self.x1_optimizer_type == other.x1_optimizer_type) & (self.x2_optimizer_lr == other.x2_optimizer_lr) & (self.x3_p_norm == other.x3_p_norm) &(self.x4_norm_flag == other.x4_norm_flag) &(self.x5_loss_type == other.x5_loss_type) &(self.x6_adv_temperature == other.x6_adv_temperature)

    def obj1_func1(self):
        p_score = self.model.getPscore()
        n_score = self.model.getNscore()
        if (n_score.mean() == 200):
            n_score = 0
        # return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
        return (p_score - n_score).mean()
        # return self.model.getPscore().mean()
        # return -(self.x1_optimizer_type + self.x2_optimizer_lr + self.x3_p_norm + self.x4_norm_flag + self.x5_loss_type + self.x6_adv_temperature)
    
    def obj2_func2(self):
        # return -self.model.getNscore().mean()
        return self.obj2        # train loss
        # return -(self.x1_optimizer_type + self.x2_optimizer_lr + self.x3_p_norm + self.x4_norm_flag - self.x5_loss_type + self.x6_adv_temperature)

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
