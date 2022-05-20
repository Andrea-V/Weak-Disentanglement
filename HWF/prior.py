import torch
import torch.nn as nn
import sklearn.datasets
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform as Unif
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.one_hot_categorical import OneHotCategorical
import numpy as np
import sys
import torch_geometric.nn as gnn
import torch_geometric
from model import MultilayerRGCN
from model import MLP
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn.functional as F

def sigmoid(x):
	return torch.sigmoid(x)
def tanh(x):
	return torch.tanh(x)
def relu(x):
	return torch.relu(x)
def linear(x):
	return x

class AdaptiveRelationalPrior(nn.Module):
		def __init__(self, cfg, z_dim, n_classes):
			super(AdaptiveRelationalPrior, self).__init__()
			self.name = "adaptive_graph"
			self.dimensions = z_dim
			self.n_classes = n_classes
			self.categorical = OneHotCategorical(probs=torch.tensor([1/n_classes for _ in range(n_classes)]))
			#self.label_fn = label_fn
			self.cfg = cfg
			gaussians = self._init_mog_random_prior(cfg)
			self.gaussians = gaussians

			self.relation_layer = MLP(
				cfg,
				n_layers=cfg.model.n_relational_layers,
				dimensions=(cfg.model.z_dim*3, 1024, cfg.model.z_dim),
				activation=torch.tanh,
				out_activation=linear
			)

		# flower non isotropic mog
		def _init_mog_flower_prior(self, cfg):
			gaussians = []
			locs = nn.ParameterList()
			scale_trils = nn.ParameterList()

			radial_noise = 0.1
			tangent_noise = 0.01

			n = cfg.data.n_classes
			d = cfg.model.z_dim

			for l in range(self.n_classes):

				# flower prior
				z_mean = torch.tensor([1 * np.cos((l*2*np.pi) / n), 1 * np.sin((l*2*np.pi) / n)] + [ 0 for _ in range(d-2) ], dtype=torch.float)
				v1 = [np.cos((l*2*np.pi) / n), np.sin((l*2*np.pi) / n)]
				v2 = [-np.sin((l*2*np.pi) / n), np.cos((l*2*np.pi) / n)]
				a1 = radial_noise   # radial axis (center-to-outer)
				a2 = tangent_noise # tangent axis (along the circle)
				M = np.eye(d)
				S = np.eye(d)
				np.fill_diagonal(S, [a1] + [ a2 for _ in range(d-1) ])
				M[0:2, 0:2] = np.vstack((v1,v2)).T
				z_cov = torch.tensor(np.dot(np.dot(M, S), np.linalg.inv(M)), dtype=torch.float)


				#p_z_mean = nn.Parameter(z_mean, requires_grad=True)
				#p_z_cov  = nn.Parameter(z_cov, requires_grad=True)

				gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)

				gaussian.loc = nn.Parameter(gaussian.loc.to(cfg.device), requires_grad=False)
				gaussian._unbroadcasted_scale_tril = nn.Parameter(gaussian._unbroadcasted_scale_tril.to(cfg.device), requires_grad=False)

				gaussians.append(gaussian)
				locs.append(gaussian.loc)
				scale_trils.append(gaussian._unbroadcasted_scale_tril)


			return gaussians, locs, scale_trils

		# adaptive mog prior
		def _init_mog_adaptive_prior(self, cfg, tr_set, model):

			if cfg.current_dataset != "HWF":
				raise ValueError("Unknown dataset: " + cfg.data.current_dataset)

			xs = []
			ys = []
			for i, (x, y, label) in enumerate(tr_set):
				if i >= cfg.prior.supervision_amount:
					break

				xs.append(x.reshape((-1, 1, 45, 45)).cpu().numpy())
				ys.append(y.reshape(-1, 1).cpu().numpy())

			x = np.array(xs).reshape((-1, 1, 45, 45))
			y = np.array(ys).reshape(-1, 1)

			gaussians = []
			for i in range(cfg.data.n_classes):
				x_i = x[y[:, 0] == i, :]

				with torch.no_grad():
					x_i = torch.Tensor(x_i).to(cfg.device)
					z_i = model.encoder(x_i).cpu().numpy()

				z_mean = np.mean(z_i, axis=0)
				z_cov  = np.cov(z_i, rowvar=False) * cfg.prior.gm_cov

				gaussians.append(
					MultivariateNormal(
						torch.Tensor(z_mean).to(cfg.device),
						torch.Tensor(z_cov).to(cfg.device)
					)
				)
			return gaussians

		# flower isotropic gauss
		def _init_mog_flower_isotropic_prior(self, cfg):
			gaussians = []
			locs = nn.ParameterList()

			n = cfg.data.n_classes
			d = cfg.model.z_dim

			for l in range(self.n_classes):
				z_mean = torch.tensor([1 * np.cos((l*2*np.pi) / n), 1 * np.sin((l*2*np.pi) / n)] + [ 0 for _ in range(d-2) ], dtype=torch.float)
				z_cov = torch.eye(d).to(cfg.device) * cfg.prior.gm_cov
				gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)
				gaussian.loc = nn.Parameter(gaussian.loc.to(cfg.device), requires_grad=True)
				gaussians.append(gaussian)
				locs.append(gaussian.loc)

			return gaussians #, locs

		# flower isotropic prior
		def _init_mog_line_prior(self, cfg):
			gaussians = []
			locs = nn.ParameterList()

			n = cfg.data.n_classes
			d = cfg.model.z_dim

			for l in range(self.n_classes):
				z_mean = torch.tensor([l, 0] + [ 0 for _ in range(d-2) ], dtype=torch.float)
				z_cov = torch.eye(d).to(cfg.device) * cfg.prior.gm_cov
				gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)
				gaussian.loc = nn.Parameter(gaussian.loc.to(cfg.device), requires_grad=True)
				gaussians.append(gaussian)
				locs.append(gaussian.loc)

			return gaussians #, locs

		# flower isotropic prior
		def _init_mog_random_prior(self, cfg):
			gaussians = []
			locs = nn.ParameterList()

			n = cfg.data.n_classes
			d = cfg.model.z_dim

			for l in range(self.n_classes):
				z_mean = (torch.rand((d,)) * 2.) - 1.
				z_cov = torch.eye(d).to(cfg.device) * cfg.prior.gm_cov
				gaussian = MultivariateNormal(loc=z_mean, covariance_matrix=z_cov)
				gaussian.loc = nn.Parameter(gaussian.loc.to(cfg.device), requires_grad=True)
				gaussians.append(gaussian)
				locs.append(gaussian.loc)

			return gaussians #, locs

		def forward(self, batch):

			add_left_select = batch.add_left_sample
			add_right_select = batch.add_right_sample
			add_relation = batch.add_relation
			add_inputs = torch.cat([add_left_select, add_right_select, add_relation], dim=-1).to(self.cfg.device)

			sub_left_select = batch.sub_left_sample
			sub_right_select = batch.sub_right_sample
			sub_relation = batch.sub_relation
			sub_inputs = torch.cat([sub_left_select, sub_right_select, sub_relation], dim=-1).to(self.cfg.device)

			mul_left_select = batch.mul_left_sample
			mul_right_select = batch.mul_right_sample
			mul_relation = batch.mul_relation
			mul_inputs = torch.cat([mul_left_select, mul_right_select, mul_relation], dim=-1).to(self.cfg.device)

			z_add_pred = self.relation_layer(add_inputs)
			z_sub_pred = self.relation_layer(sub_inputs)
			z_mul_pred = self.relation_layer(mul_inputs)

			return z_add_pred, z_sub_pred, z_mul_pred

		def sample(self, y):
			z_sample = torch.empty((self.cfg.data.batch_size, self.dimensions))

			y_sample = y

			for i, y_sample_i in enumerate(y_sample):
				z_sample[i, :] = self.gaussians[y_sample_i.argmax()].sample()

			z_sample = z_sample.to(self.cfg.device)
			#y_sample = y_sample.to(self.cfg.device)

			return z_sample

		def classify(self, z, threshold=None):
			log_threshold = torch.log(torch.Tensor([threshold]).to(self.cfg.device))

			# classify only on the entites gaussians
			log_probs = torch.stack([ self.gaussians[i].log_prob(z) for i in range(self.cfg.data.n_entities)], dim=-1).to(self.cfg.device)

			max_val, max_i = log_probs.max(dim=-1)
			max_i[max_val < log_threshold ] = -1

			return max_i

		def init_adaptive_prior(self, tr_set, model):
			gaussians = self._init_mog_adaptive_prior(self.cfg, tr_set, model)
			self.gaussians = gaussians


class MixtureOfGaussianPrior(nn.Module):
	def __init__(self, z_dim, means, covariances):
		super(MixtureOfGaussianPrior, self).__init__()
		self.name = "MixtureOfGaussian"
		self.dimensions = z_dim
		self.gaussians = []
		n_gauss = len(means)
		self.categorical = OneHotCategorical(probs=torch.tensor([1/n_gauss for _ in range(n_gauss)]))

		assert len(means) == len(covariances)

		for z_mean, z_cov in zip(means, covariances):
			self.gaussians.append(
				MultivariateNormal(
					torch.Tensor(z_mean).to(self.cfg.device),
					torch.Tensor(z_cov).to(self.cfg.device)
				)
			)

	def forward(self):
		z_sample = torch.empty((self.cfg.data.batch_size, self.dimensions))
		y_sample = self.categorical.sample((self.cfg.data.batch_size,))

		for i, y_sample_i in enumerate(y_sample):
			z_sample[i, :] = self.gaussians[y_sample_i.argmax()].sample()

		z_sample = z_sample.to(self.cfg.device)
		y_sample = y_sample.to(self.cfg.device)
		return y_sample, z_sample

	def classify(self, z, threshold=None):
		log_threshold = torch.log(torch.Tensor([threshold]).to(self.cfg.device))
		log_probs = torch.stack([ gaussian.log_prob(z) for gaussian in self.gaussians], dim=-1).to(self.cfg.device)
		max_val, max_i = log_probs.max(dim=-1)
		max_i[max_val < log_threshold ] = -1
		return max_i


class DoubleMoon(nn.Module):
	def __init__(self, noise):
		super(DoubleMoon, self).__init__()
		self.name = "double_moon"
		self.noise_p = nn.Parameter(torch.tensor(noise))
		self.noise = noise

	def forward(self, y):
		x, _ = sklearn.datasets.make_moons(n_samples=y.shape[0], shuffle=True, noise=self.noise)#self.noise.item())
		return torch.tensor(x).float().to(config.device)

class SwissRoll(nn.Module):
	def __init__(self, cfg, noise):
		super(SwissRoll, self).__init__()
		self.name = "swiss_roll"
		self.noise_p = nn.Parameter(torch.tensor(noise))
		self.noise = noise
		self.cfg = cfg

	def forward(self, y):
		x, _ = sklearn.datasets.make_swiss_roll(n_samples=self.cfg.data.batch_size, noise=self.noise)
		x = x.T[[0, 2], :].T # remove 2nd dimension

		return torch.tensor(x).float().to(self.cfg.device)

class Flower(nn.Module):
	def __init__(self, cfg, n_gaussians, dimensions, radial_noise=0.1, tangent_noise=0.001):
		super(Flower, self).__init__()
		self.cfg = cfg
		self.name = "flower"
		n = n_gaussians 
		d = dimensions
		self.n_p = nn.Parameter(torch.tensor(float(n)))

		self.radial_noise = radial_noise
		self.tangent_noise = tangent_noise
		self.dimensions = d
		self.n_gaussians = n
		self.n_classes = n
		self.prior_mean = np.empty(shape=(n, d))
		self.prior_cov = np.empty(shape=(n, d, d))

		for l in range(n):
			self.prior_mean[l] = [1 * np.cos((l*2*np.pi) / n), 1 * np.sin((l*2*np.pi) / n)] + [ 0 for _ in range(d-2) ]
			v1 = [np.cos((l*2*np.pi) / n), np.sin((l*2*np.pi) / n)]
			v2 = [-np.sin((l*2*np.pi) / n), np.cos((l*2*np.pi) / n)]
			a1 = radial_noise   # radial axis (center-to-outer)
			a2 = tangent_noise # tangent axis (along the circle)  
			M = np.eye(d)
			S = np.eye(d)
			np.fill_diagonal(S, [a1] + [ a2 for _ in range(d-1) ])
			M[0:2, 0:2] = np.vstack((v1,v2)).T
			self.prior_cov[l] = np.dot(np.dot(M, S), np.linalg.inv(M))

		self.gaussians = [
			MultivariateNormal(
				torch.tensor(self.prior_mean[l]).float().to(cfg.device),
				torch.tensor(self.prior_cov[l]).float().to(cfg.device)
			)
			for l in range(n)
		]

		self.categorical = OneHotCategorical(probs=torch.tensor([1/n for _ in range(n)]))

	def forward(self, y):
		z_sample = torch.empty((self.cfg.data.batch_size, self.dimensions))
		#y_sample = self.categorical.sample((self.cfg.data.batch_size,))
		y_sample = y

		for i, y_sample_i in enumerate(y_sample):
			z_sample[i, :] = self.gaussians[y_sample_i.argmax()].sample()		
		
		z_sample = z_sample.to(self.cfg.device)
		y_sample = y_sample.to(self.cfg.device)

		return z_sample

	
	def classify(self, z, threshold=None):
		log_threshold = torch.log(torch.Tensor([threshold]).to(self.cfg.device))
		
		# print("z:", z)
		# print("z:", z.shape)

		log_probs =torch.stack([ gaussian.log_prob(z) for gaussian in self.gaussians], dim=-1).to(self.cfg.device)

		max_val, max_i = log_probs.max(dim=-1)

		max_i[max_val < log_threshold ] = -1

		return max_i

class MixtureOf2Gaussians(nn.Module):
	def __init__(self, cfg, z_dim, mu1, sigma1, mu2, sigma2):
		super(MixtureOf2Gaussians, self).__init__()
		self.name = "mixture_of_2gauss"		
		self.cfg = cfg
		self.z_dim = z_dim
		# not actually needeed, just so that the prior optim doesn't get an empty list of parameters
		self.mu1 = nn.Parameter(torch.tensor(mu1))
		self.sigma1 = nn.Parameter(torch.tensor(sigma1))
		self.mu2 = nn.Parameter(torch.tensor(mu2))
		self.sigma2 = nn.Parameter(torch.tensor(sigma2))
		
		self.p1 = MultivariateNormal(torch.tensor(mu1), torch.tensor(sigma1))
		self.p2 = MultivariateNormal(torch.tensor(mu2), torch.tensor(sigma2))
		self.categorical = OneHotCategorical(probs=torch.tensor([1/self.cfg.data.n_classes for _ in range(self.cfg.data.n_classes)]))


	def forward(self, y):
		y_sample = y
		#y_sample = self.categorical.sample((config.data.batch_size,)).to(config.device)
		y = y_sample.argmax(dim=-1)

		sample = torch.zeros((y.shape[0], self.z_dim)).to(self.cfg.device)
		sample1 = self.p1.sample((y.shape[0],)).to(self.cfg.device)
		sample2 = self.p2.sample((y.shape[0],)).to(self.cfg.device)
		sample[y <= 4] = sample1[y <= 4]
		sample[y > 4] = sample2[y > 4]
		return y_sample, sample

	def sample(self, y):
		y_sample = y
		#y_sample = self.categorical.sample((config.data.batch_size,)).to(config.device)
		y = y_sample.argmax(dim=-1)

		sample = torch.zeros((y.shape[0], self.z_dim)).to(self.cfg.device)
		sample1 = self.p1.sample((y.shape[0],)).to(self.cfg.device)
		sample2 = self.p2.sample((y.shape[0],)).to(self.cfg.device)
		sample[y <= 4] = sample1[y <= 4]
		sample[y > 4] = sample2[y > 4]
		return sample

class Gaussian(nn.Module):
	def __init__(self, cfg, z_dim, mu, sigma):
		super(Gaussian, self).__init__()
		self.name = "gaussian"
		# not actually needeed, just so that the prior optim doesn't get an empty list of parameters
		self.mu = nn.Parameter(torch.tensor(mu))
		self.sigma = nn.Parameter(torch.tensor(sigma))
		self.cfg = cfg
		self.p = Normal(torch.tensor([mu]*z_dim), torch.tensor([sigma]*z_dim))
		self.categorical = OneHotCategorical(probs=torch.tensor([1. for _ in range(1)]))

	def forward(self, n_samples=None):
		if not n_samples:
			n_samples = self.cfg.data.batch_size
		y_sample = self.categorical.sample((n_samples,)).to(self.cfg.device)
		sample = self.p.sample((n_samples,)).to(self.cfg.device)
		return y_sample, sample

class Uniform(nn.Module): # isotropic uniform dist
	def __init__(self, cfg, z_dim, low, high):
		super(Uniform, self).__init__()
		self.cfg = cfg
		self.name = "uniform"
		# not actually needeed, just so that the prior optim doesn't get an empty list of parameters
		self.low = nn.Parameter(torch.tensor(low))
		self.high = nn.Parameter(torch.tensor(high))

		self.p = Unif(torch.tensor([low]*z_dim), torch.tensor([high]*z_dim))

		self.categorical = OneHotCategorical(probs=torch.tensor([1/self.cfg.data.n_classes for _ in range(self.cfg.data.n_classes)]))

	def forward(self):
		#y_sample = self.categorical.sample((self.cfg.data.batch_size,)).to(self.cfg.device)
		sample = self.p.sample((self.cfg.data.batch_size,)).to(self.cfg.device)
		return sample

	def sample(self, y):
		#y_sample = self.categorical.sample((self.cfg.data.batch_size,)).to(self.cfg.device)
		sample = self.p.sample((self.cfg.data.batch_size,)).to(self.cfg.device)
		return  sample
