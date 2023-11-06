import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from .Loss import Loss

class MarginLoss(Loss):

	def __init__(self, adv_temperature = 0, margin = 6.0):
		super(MarginLoss, self).__init__()
		self.margin = nn.Parameter(torch.Tensor([margin]))
		self.margin.requires_grad = False
		self.adv_temperature_value = adv_temperature
		if adv_temperature != 0:
			self.adv_temperature = nn.Parameter(torch.Tensor([adv_temperature]))
			self.adv_temperature.requires_grad = False
			self.adv_flag = True
		else:
			self.adv_flag = False

	def getType(self):
		return 'marginloss'

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
		return F.softmax(-n_score * self.adv_temperature, dim = -1).detach()

	def forward(self, p_score, n_score):
		if self.adv_flag:
			return (self.get_weights(n_score) * torch.max(p_score - n_score, -self.margin)).sum(dim = -1).mean() + self.margin
		else:
			return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin
			
	
	def predict(self, p_score, n_score):
		score = self.forward(p_score, n_score)
		return score.cpu().data.numpy()