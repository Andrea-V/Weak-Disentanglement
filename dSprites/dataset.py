from torchvision import transforms
from torch.utils.data import Dataset, Sampler
import torch
import torch.utils.data
import json
import os
import random
import numpy as np
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional
from PIL import Image
import tqdm
import torch_geometric
import sys
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt

class UnbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	def __init__(self, indexes, weights, num_samples, replacement):
		super(UnbalancedDatasetSampler, self).__init__(data_source=indexes)
		self.indexes = indexes
		self.weights = weights
		self.weighted_sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=num_samples, replacement=replacement)

	def __iter__(self):
		return ( self.indexes[sample] for sample in self.weighted_sampler )

	def __len__(self):
		return len(self.weighted_sampler)

class dSprites(Dataset):
	def __init__(self, cfg, dataset_path):
		super(dSprites, self).__init__()
		self.dsprites = np.load(dataset_path, allow_pickle=True)
		self.cfg = cfg
		self.xs = []
		self.ys = []
		
		for x, y in zip(self.dsprites["imgs"], self.dsprites["latents_classes"]):
			x_pos_left   = abs(y[4]-0) <= 4
			x_pos_center = abs(y[4]-15) <= 2
			x_pos_right  = abs(y[4]-31) <= 4

			y_pos_up     = abs(y[5]-0) <= 4
			y_pos_center = abs(y[5]-15) <= 2
			y_pos_down   = abs(y[5]-31) <= 4

			orientation_condition = abs(y[3]-0) <= 2 or abs(y[3]-39) <= 2

			scale_condition = abs(y[2]-3) <= 1
			x_pos_condition = x_pos_left or x_pos_center or x_pos_right
			y_pos_condition = y_pos_up or y_pos_center or y_pos_down

			if x_pos_condition and y_pos_condition and scale_condition and orientation_condition:

				if x_pos_left:
					y[4] = 0
				if x_pos_center:
					y[4] = 1
				if x_pos_right:
					y[4] = 2
				if y_pos_up:
					y[5] = 0
				if y_pos_center:
					y[5] = 1
				if y_pos_down:
					y[5] = 2

				x = torch.Tensor(x).float()
				y = torch.Tensor(y).long()

				self.xs.append(x.clone())
				self.ys.append(y.clone())

				n_rep = 3
				for _ in range(n_rep):
					noise = x + torch.randn_like(x) * 0.3

					x3 = x.clone()
					x3[x == 1] = noise[x == 1]
					x3[x3 > 1] = 1
					x3[x3 < 0] = 0
					self.xs.append(x3.clone())
					self.ys.append(y.clone())

				for _ in range(n_rep):
					x4 = x.clone()
					mask = torch.bernoulli(torch.full_like(x, 0.3))
					x4[mask == 1] = 0
					self.xs.append(x4.clone())
					self.ys.append(y.clone())

	def _from_label_to_1hot(self, label):

		shape = label[1] * 9
		x_pos = label[4] * 3
		y_pos = label[5]
		hot_idx = shape + x_pos + y_pos

		#print(label, "->", hot_idx)

		hot_label = torch.zeros((self.cfg.data.n_entities,))
		hot_label[hot_idx] = 1
		return hot_label

	def __len__(self):
		return len(self.ys)

	# the item MUST NOT have batch dimension
	def __getitem__(self, idx):
		y = self.ys[idx]
		return self.xs[idx], self._from_label_to_1hot(y), y


def load_dsprites(cfg, path, batch_size):
	dsprites = dSprites(cfg, path)

	idxs = list(range(len(dsprites)))

	random.shuffle(idxs)
	ten_percent = int(0.1 * len(idxs))
	tr_idxs = idxs[:-2*ten_percent]
	vl_idxs = idxs[-2*ten_percent:-ten_percent]
	ts_idxs = idxs[-ten_percent:]

	tr_sampler = torch.utils.data.SubsetRandomSampler(tr_idxs)
	vl_sampler = torch.utils.data.SubsetRandomSampler(vl_idxs)
	ts_sampler = torch.utils.data.SubsetRandomSampler(ts_idxs)

	tr_loader = torch.utils.data.DataLoader(
		dsprites,
		batch_size=batch_size,
		sampler=tr_sampler,
		num_workers=cfg.data.n_workers_tr,
		drop_last=False
	)

	vl_loader = torch.utils.data.DataLoader(
		dsprites,
		batch_size=batch_size,
		sampler=vl_sampler,
		num_workers=cfg.data.n_workers_vl,
		drop_last=False
	)

	ts_loader = torch.utils.data.DataLoader(
		dsprites,
		batch_size=batch_size,
		sampler=ts_sampler,
		num_workers=cfg.data.n_workers_ts,
		drop_last=False
	)

	return tr_loader, vl_loader, ts_loader

def _build_dsprites_relations(cfg, prior, digits_node_id, fix=None):

	def compute_left_target(start_idx):
		table = {
            "<^S":">^S", "<xS":">xS", "<vS":">vS", "x^S":"<^S", "xxS":"<xS", "xvS":"<vS", ">^S":"x^S", ">xS":"xxS", ">vS":"xvS",
            "<^E":">^E", "<xE":">xE", "<vE":">vE", "x^E":"<^E", "xxE":"<xE", "xvE":"<vE", ">^E":"x^E", ">xE":"xxE", ">vE":"xvE",
            "<^H":">^H", "<xH":">xH", "<vH":">vH", "x^H":"<^H", "xxH":"<xH", "xvH":"<vH", ">^H":"x^H", ">xH":"xxH", ">vH":"xvH",
		}
		start = cfg.data.class_names[start_idx]
		target = table[start]
		target_idx = cfg.data.class_names.index(target)
		return target_idx

	def compute_right_target(start_idx):
		table = {
            "<^S":"x^S", "<xS":"xxS", "<vS":"xvS", "x^S":">^S", "xxS":">xS", "xvS":">vS", ">^S":"<^S", ">xS":"<xS", ">vS":"<vS",
            "<^E":"x^E", "<xE":"xxE", "<vE":"xvE", "x^E":">^E", "xxE":">xE", "xvE":">vE", ">^E":"<^E", ">xE":"<xE", ">vE":"<vE",
            "<^H":"x^H", "<xH":"xxH", "<vH":"xvH", "x^H":">^H", "xxH":">xH", "xvH":">vH", ">^H":"<^H", ">xH":"<xH", ">vH":"<vH",
		}
		start = cfg.data.class_names[start_idx]
		target = table[start]

		target_idx = cfg.data.class_names.index(target)
		return target_idx

	def compute_up_target(start_idx):
		table = {
            "<^S":"<vS", "<xS":"<^S", "<vS":"<xS", "x^S":"xvS", "xxS":"x^S", "xvS":"xxS", ">^S":">vS", ">xS":">^S", ">vS":">xS",
            "<^E":"<vE", "<xE":"<^E", "<vE":"<xE", "x^E":"xvE", "xxE":"x^E", "xvE":"xxE", ">^E":">vE", ">xE":">^E", ">vE":">xE",
            "<^H":"<vH", "<xH":"<^H", "<vH":"<xH", "x^H":"xvH", "xxH":"x^H", "xvH":"xxH", ">^H":">vH", ">xH":">^H", ">vH":">xH",
		}
		start = cfg.data.class_names[start_idx]
		target = table[start]
		target_idx = cfg.data.class_names.index(target)
		return target_idx


	def compute_down_target(start_idx):
		table = {
            "<^S":"<xS", "<xS":"<vS", "<vS":"<^S", "x^S":"xxS", "xxS":"xvS", "xvS":"x^S", ">^S":">xS", ">xS":">vS", ">vS":">^S",
            "<^E":"<xE", "<xE":"<vE", "<vE":"<^E", "x^E":"xxE", "xxE":"xvE", "xvE":"x^E", ">^E":">xE", ">xE":">vE", ">vE":">^E",
            "<^H":"<xH", "<xH":"<vH", "<vH":"<^H", "x^H":"xxH", "xxH":"xvH", "xvH":"x^H", ">^H":">xH", ">xH":">vH", ">vH":">^H",
		}
		start = cfg.data.class_names[start_idx]
		target = table[start]
		target_idx = cfg.data.class_names.index(target)
		return target_idx


	def compute_shape_target(start_idx):
		table = {
            "<^S":"<^E", "<xS":"<xE", "<vS":"<vE", "x^S":"x^E", "xxS":"xxE", "xvS":"xvE", ">^S":">^E", ">xS":">xE", ">vS":">vE",
            "<^E":"<^H", "<xE":"<xH", "<vE":"<vH", "x^E":"x^H", "xxE":"xxH", "xvE":"xvH", ">^E":">^H", ">xE":">xH", ">vE":">vH",
            "<^H":"<^S", "<xH":"<xS", "<vH":"<vS", "x^H":"x^S", "xxH":"xxS", "xvH":"xvS", ">^H":">^S", ">xH":">xS", ">vH":">vS"
		}
		start = cfg.data.class_names[start_idx]
		target = table[start]
		target_idx = cfg.data.class_names.index(target)
		return target_idx

	if fix is not None:
		assert type(fix) == int
		left = fix
		left_sample = prior.gaussians[fix].rsample()
		right = fix
		right_sample = prior.gaussians[fix].rsample()
		up = fix
		up_sample = prior.gaussians[fix].rsample()
		down = fix
		down_sample = prior.gaussians[fix].rsample()
		shape = fix
		shape_sample = prior.gaussians[fix].rsample()
	else:
		left = np.random.choice(digits_node_id)
		left_sample = prior.gaussians[left].rsample()
		right = np.random.choice(digits_node_id)
		right_sample = prior.gaussians[right].rsample()
		up = np.random.choice(digits_node_id)
		up_sample = prior.gaussians[up].rsample()
		down = np.random.choice(digits_node_id)
		down_sample = prior.gaussians[down].rsample()
		shape = np.random.choice(digits_node_id)
		shape_sample = prior.gaussians[shape].rsample()

	left_target = compute_left_target(left)
	left_target_sample = prior.gaussians[left_target].rsample()

	right_target = compute_right_target(right)
	right_target_sample = prior.gaussians[right_target].rsample()

	up_target = compute_up_target(up)
	up_target_sample = prior.gaussians[up_target].rsample()

	down_target = compute_down_target(down)
	down_target_sample = prior.gaussians[down_target].rsample()

	shape_target = compute_shape_target(shape)
	shape_target_sample = prior.gaussians[shape_target].rsample()

	left_data = DSprites_RelationalData(
		label=torch.tensor([left], dtype=torch.long),
		sample=left_sample,
		target_sample=left_target_sample,
		target_label=torch.tensor([left_target], dtype=torch.long)
	)
	right_data = DSprites_RelationalData(
		label=torch.tensor([right], dtype=torch.long),
		sample=right_sample,
		target_sample=right_target_sample,
		target_label=torch.tensor([right_target], dtype=torch.long)
	)
	up_data = DSprites_RelationalData(
		label=torch.tensor([up], dtype=torch.long),
		sample=up_sample,
		target_sample=up_target_sample,
		target_label=torch.tensor([up_target], dtype=torch.long)
	)
	down_data = DSprites_RelationalData(
		label=torch.tensor([down], dtype=torch.long),
		sample=down_sample,
		target_sample=down_target_sample,
		target_label=torch.tensor([down_target], dtype=torch.long)
	)
	shape_data = DSprites_RelationalData(
		label=torch.tensor([shape], dtype=torch.long),
		sample=shape_sample,
		target_sample=shape_target_sample,
		target_label=torch.tensor([shape_target], dtype=torch.long)
	)
	return left_data, right_data, up_data, down_data, shape_data

def new_batch_of_dsprites_relations(cfg, batch_size, prior, digits_node_id, fix_left=None):

	left_relations = []
	right_relations = []
	up_relations = []
	down_relations = []
	shape_relations = []

	for _ in range(batch_size):
		lr, rr, ur, dr, sr = _build_dsprites_relations(cfg, prior, digits_node_id, fix_left)
		left_relations.append(lr)
		right_relations.append(rr)
		up_relations.append(ur)
		down_relations.append(dr)
		shape_relations.append(sr)

	batch = DSprites_RelationalBatch(cfg, left_relations, right_relations, up_relations, down_relations, shape_relations)
	return batch


class DSprites_RelationalData():
	def __init__(self, label, sample, target_label, target_sample):
		super(DSprites_RelationalData, self).__init__()
		self.label = label
		self.sample = sample
		self.target_sample = target_sample
		self.target_label = target_label

class DSprites_RelationalBatch():
	def __init__(self, cfg, left_relations, right_relations, up_relations, down_relations, shape_relations):
		super(DSprites_RelationalBatch, self).__init__()
		self.z_dim = cfg.model.z_dim

		self.left_label = []
		self.left_sample = []
		self.left_target_label = []
		self.left_target_sample = []

		for i, data in enumerate(left_relations):
			self.left_label.append(data.label)
			self.left_sample.append(data.sample)
			self.left_target_label.append(data.target_label)
			self.left_target_sample.append(data.target_sample)

		self.left_label = torch.cat(self.left_label, dim=0)
		self.left_sample = torch.vstack(self.left_sample)
		self.left_target_label = torch.cat(self.left_target_label, dim=0)
		self.left_target_sample = torch.vstack(self.left_target_sample)

		self.right_label = []
		self.right_sample = []
		self.right_target_label = []
		self.right_target_sample = []

		for i, data in enumerate(right_relations):
			self.right_label.append(data.label)
			self.right_sample.append(data.sample)
			self.right_target_label.append(data.target_label)
			self.right_target_sample.append(data.target_sample)

		self.right_label = torch.cat(self.right_label, dim=0)
		self.right_sample = torch.vstack(self.right_sample)
		self.right_target_label = torch.cat(self.right_target_label, dim=0)
		self.right_target_sample = torch.vstack(self.right_target_sample)

		self.up_label = []
		self.up_sample = []
		self.up_target_label = []
		self.up_target_sample = []

		for i, data in enumerate(up_relations):
			self.up_label.append(data.label)
			self.up_sample.append(data.sample)
			self.up_target_label.append(data.target_label)
			self.up_target_sample.append(data.target_sample)

		self.up_label = torch.cat(self.up_label, dim=0)
		self.up_sample = torch.vstack(self.up_sample)
		self.up_target_label = torch.cat(self.up_target_label, dim=0)
		self.up_target_sample = torch.vstack(self.up_target_sample)

		self.down_label = []
		self.down_sample = []
		self.down_target_label = []
		self.down_target_sample = []

		for i, data in enumerate(down_relations):
			self.down_label.append(data.label)
			self.down_sample.append(data.sample)
			self.down_target_label.append(data.target_label)
			self.down_target_sample.append(data.target_sample)

		self.down_label = torch.cat(self.down_label, dim=0)
		self.down_sample = torch.vstack(self.down_sample)
		self.down_target_label = torch.cat(self.down_target_label, dim=0)
		self.down_target_sample = torch.vstack(self.down_target_sample)

		self.shape_label = []
		self.shape_sample = []
		self.shape_target_label = []
		self.shape_target_sample = []

		for i, data in enumerate(shape_relations):
			self.shape_label.append(data.label)
			self.shape_sample.append(data.sample)
			self.shape_target_label.append(data.target_label)
			self.shape_target_sample.append(data.target_sample)

		self.shape_label = torch.cat(self.shape_label, dim=0)
		self.shape_sample = torch.vstack(self.shape_sample)
		self.shape_target_label = torch.cat(self.shape_target_label, dim=0)
		self.shape_target_sample = torch.vstack(self.shape_target_sample)


	def to(self, device):
		self.left_label = self.left_label.to(device)
		self.left_sample = self.left_sample.to(device)
		self.left_target_label = self.left_target_label.to(device)
		self.left_target_sample = self.left_target_sample.to(device)

		self.right_label = self.right_label.to(device)
		self.right_sample = self.right_sample.to(device)
		self.right_target_label = self.right_target_label.to(device)
		self.right_target_sample = self.right_target_sample.to(device)

		self.up_label = self.up_label.to(device)
		self.up_sample = self.up_sample.to(device)
		self.up_target_label = self.up_target_label.to(device)
		self.up_target_sample = self.up_target_sample.to(device)

		self.down_label = self.down_label.to(device)
		self.down_sample = self.down_sample.to(device)
		self.down_target_label = self.down_target_label.to(device)
		self.down_target_sample = self.down_target_sample.to(device)

		self.shape_label = self.shape_label.to(device)
		self.shape_sample = self.shape_sample.to(device)
		self.shape_target_label = self.shape_target_label.to(device)
		self.shape_target_sample = self.shape_target_sample.to(device)

		return self
