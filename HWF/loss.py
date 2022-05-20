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
		loss = F.mse_loss(z_true, z_pred)

		return loss

class RelationalLossCompute(nn.Module):
	def __init__(self, cfg,  criterion, optim, train):
		super(RelationalLossCompute, self).__init__()
		self.train = train
		self.criterion = criterion
		self.optim = optim
		self.cfg = cfg

	def __call__(self, z_true_add, z_pred_add, z_true_sub, z_pred_sub, z_true_mul, z_pred_mul, retain_graph=False):

		loss = (self.criterion(z_true_add, z_pred_add) + self.criterion(z_true_sub, z_pred_sub) + self.criterion(z_true_mul, z_pred_mul)) / 3.
		#loss = self.criterion(z_true, z_pred)
		loss = self.cfg.loss.prior_weight * loss

		if self.train:
			self.optim.zero_grad()
			loss.backward(retain_graph=retain_graph)
			self.optim.step()

		return loss


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

		return loss, None # None is grad penalty (to have interface consistent with wgan)

class GeneratorLossCompute(nn.Module):
	def __init__(self, cfg, criterion, optimizer, train):
		super(GeneratorLossCompute, self).__init__()
		self.criterion = criterion
		self.train = train
		self.optimizer = optimizer
		self.cfg = cfg

	def __call__(self, z_gen_score, retain=False):
		
		#loss = self.criterion(z_gen_score, torch.tensor([1] * z_gen_score.shape[0]).to(config.device))
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

