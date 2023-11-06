import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Loss import Loss

class SigmoidLoss(Loss):

	def __init__(self, adv_temperature = 0):
		super(SigmoidLoss, self).__init__()
		self.criterion = nn.LogSigmoid()
		self.adv_temperature_value = adv_temperature
		if adv_temperature != 0:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False
	
	def getType(self):
		return 'sigmoidloss'

	def getAdvTemperature(self):
		return self.adv_temperature_value

	def setAdvTemperature(self, value):
		self.adv_temperature_value = value
		if self.adv_temperature_value != 0:
			self.adv_temperature = nn.Parameter(torch.Tensor([self.adv_temperature_value]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False

	def get_weights(self, n_score):
		return F.softmax(n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return -(self.criterion(p_score).mean() + (self.get_weights(n_score) * self.criterion(-n_score)).sum(dim = -1).mean()) / 2
		else:
			return -(self.criterion(p_score).mean() + self.criterion(-n_score).mean()) / 2

	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()