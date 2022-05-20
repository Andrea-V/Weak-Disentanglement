import torch
import torch.nn as nn
from torch.functional import F
import numpy as np
import torch.autograd
import sys

class RelationalLoss(nn.Module):
	def __init__(self, prior):
		super(RelationalLoss, self).__init__()
		self.prior = prior

	def __call__(self, z_true, z_pred):
		loss = F.mse_loss(z_pred, z_true)
		return loss

class RelationalLossCompute(nn.Module):
	def __init__(self, cfg,  criterion, optim, train):
		super(RelationalLossCompute, self).__init__()
		self.train = train
		self.criterion = criterion
		self.optim = optim
		self.cfg = cfg

	def __call__(self, z_true_left, z_true_right, z_true_up, z_true_down, z_true_shape, z_left, z_right, z_up, z_down, z_shape, retain_graph=False):

		loss_left = self.criterion(z_true_left, z_left)
		loss_right = self.criterion(z_true_right, z_right)
		loss_up = self.criterion(z_true_up, z_up)
		loss_down = self.criterion(z_true_down, z_down)
		loss_shape = self.criterion(z_true_shape, z_shape)

		loss = self.cfg.loss.prior_weight * (loss_left + loss_right + loss_up + loss_down + loss_shape) / 5.

		if self.train:
			self.optim.zero_grad()
			loss.backward(retain_graph=retain_graph)
			self.optim.step()

		return loss

# vanilla GAN LOSS COMPUTE
class DiscriminatorLossCompute(nn.Module):
	def __init__(self, cfg, criterion, optimizer, train):
		super(DiscriminatorLossCompute, self).__init__()
		self.criterion = criterion
		self.train = train
		self.optimizer = optimizer
		self.cfg = cfg

	def __call__(self, z_gen_score, z_prior_score, z_gen, z_prior, retain=False):

		loss_gen   = self.criterion(z_gen_score, torch.zeros_like(z_gen_score).to(self.cfg.device))
		loss_prior = self.criterion(z_prior_score, torch.ones_like(z_prior_score).to(self.cfg.device))
		
		loss =  .5 * loss_gen + .5 * loss_prior
		loss = self.cfg.loss.adversarial_weight * loss.mean()

		if self.train:
			self.optimizer.zero_grad()
			loss.backward(retain_graph=retain)
			#nn.utils.clip_grad_value_(self.optimizer.param_groups[0]["params"], config.train.clip_value)
			self.optimizer.step()

		return loss, None 

class GeneratorLossCompute(nn.Module):
	def __init__(self, cfg, criterion, optimizer, train):
		super(GeneratorLossCompute, self).__init__()
		self.criterion = criterion
		self.train = train
		self.optimizer = optimizer
		self.cfg = cfg

	def __call__(self, z_gen_score, retain=False):
		
		loss = self.criterion(z_gen_score, torch.ones_like(z_gen_score).to(self.cfg.device))
		loss = self.cfg.loss.adversarial_weight * loss.mean()
		
		if self.train:
			self.optimizer.zero_grad()
			loss.backward(retain_graph=retain)
			#nn.utils.clip_grad_value_(self.optimizer.param_groups[0]["params"], config.train.clip_value)
			self.optimizer.step()

		return loss

class ReconstructionLossCompute(nn.Module):
	def __init__(self, cfg, criterion, enc_optim, dec_optim, train):
		super(ReconstructionLossCompute, self).__init__()
		self.criterion = criterion
		self.train = train
		self.enc_optim = enc_optim
		self.dec_optim = dec_optim
		self.cfg = cfg

	def __call__(self, x_rec, x, retain_graph=False):
		loss = self.criterion(x_rec, x)
		if self.train:
			self.enc_optim.zero_grad()
			self.dec_optim.zero_grad()
			loss.backward(retain_graph=retain_graph)
			#nn.utils.clip_grad_value_(self.optimizer.param_groups[0]["params"], config.train.clip_value)
			self.enc_optim.step()
			self.dec_optim.step()

		return loss

