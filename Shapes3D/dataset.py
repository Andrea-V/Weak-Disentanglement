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

class Shapes3D(Dataset):
	def __init__(self, cfg, dataset_path):
		super(Shapes3D, self).__init__()
		self.Shapes3D = np.load(dataset_path, allow_pickle=True)
		self.cfg = cfg
		self.xs = []
		self.ys = []


		for x, y in zip(self.Shapes3D["imgs"], self.Shapes3D["latents_classes"]):
			x_pos_more_hue   = abs(y[4]-0) <= 4
			x_pos_center = abs(y[4]-15) <= 2
			x_pos_less_hue  = abs(y[4]-31) <= 4

			y_pos_bigger     = abs(y[5]-0) <= 4
			y_pos_center = abs(y[5]-15) <= 2
			y_pos_smaller   = abs(y[5]-31) <= 4

			orientation_condition = abs(y[3]-0) <= 2 or abs(y[3]-39) <= 2

			scale_condition = abs(y[2]-3) <= 1
			x_pos_condition = x_pos_more_hue or x_pos_center or x_pos_less_hue
			y_pos_condition = y_pos_bigger or y_pos_center or y_pos_smaller

			if x_pos_condition and y_pos_condition and scale_condition and orientation_condition:

				if x_pos_more_hue:
					y[4] = 0
				if x_pos_center:
					y[4] = 1
				if x_pos_less_hue:
					y[4] = 2
				if y_pos_bigger:
					y[5] = 0
				if y_pos_center:
					y[5] = 1
				if y_pos_smaller:
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


def load_Shapes3D(cfg, path, batch_size):
	Shapes3D = Shapes3D(cfg, path)

	idxs = list(range(len(Shapes3D)))
	random.shuffle(idxs)
	ten_percent = int(0.1 * len(idxs))
	tr_idxs = idxs[:-2*ten_percent]
	vl_idxs = idxs[-2*ten_percent:-ten_percent]
	ts_idxs = idxs[-ten_percent:]

	tr_sampler = torch.utils.data.SubsetRandomSampler(tr_idxs)
	vl_sampler = torch.utils.data.SubsetRandomSampler(vl_idxs)
	ts_sampler = torch.utils.data.SubsetRandomSampler(ts_idxs)

	tr_loader = torch.utils.data.DataLoader(
		Shapes3D,
		batch_size=batch_size,
		sampler=tr_sampler,
		num_workers=cfg.data.n_workers_tr,
		drop_last=False
	)

	vl_loader = torch.utils.data.DataLoader(
		Shapes3D,
		batch_size=batch_size,
		sampler=vl_sampler,
		num_workers=cfg.data.n_workers_vl,
		drop_last=False
	)

	ts_loader = torch.utils.data.DataLoader(
		Shapes3D,
		batch_size=batch_size,
		sampler=ts_sampler,
		num_workers=cfg.data.n_workers_ts,
		drop_last=False
	)

	return tr_loader, vl_loader, ts_loader

def _build_Shapes3D_relations(cfg, prior, digits_node_id, fix=None):

	def compute_more_hue_target(start_idx):
		def table(start):
			hue, shape, scale = start.split(",")
			hue_v   = int(hue[-1])
			shape_v = int(shape[-1])
			scale_v = int(scale[-1])

			if hue_v < 9:
				hue_v += 1

			return "HU" + str(hue_v) + ",SH" + str(shape_v) + ",SC" + str(scale_v)

		start = cfg.data.class_names[start_idx]
		target = table(start)
		target_idx = cfg.data.class_names.index(target)
		return target_idx

	def compute_less_hue_target(start_idx):

		def table(start):
			hue, shape, scale = start.split(",")
			hue_v   = int(hue[-1])
			shape_v = int(shape[-1])
			scale_v = int(scale[-1])

			if hue_v > 0:
				hue_v -= 1

			return "HU" + str(hue_v) + ",SH" + str(shape_v) + ",SC" + str(scale_v)

		start = cfg.data.class_names[start_idx]
		target = table[start]

		target_idx = cfg.data.class_names.index(target)
		return target_idx

	def compute_bigger_target(start_idx):
		def table(start):
			hue, shape, scale = start.split(",")
			hue_v   = int(hue[-1])
			shape_v = int(shape[-1])
			scale_v = int(scale[-1])

			if scale_v < 2:
				scale_v += 1

			return "HU" + str(hue_v) + ",SH" + str(shape_v) + ",SC" + str(scale_v)

		start = cfg.data.class_names[start_idx]
		target = table[start]
		target_idx = cfg.data.class_names.index(target)
		return target_idx


	def compute_smaller_target(start_idx):
		def table(start):
			hue, shape, scale = start.split(",")
			hue_v   = int(hue[-1])
			shape_v = int(shape[-1])
			scale_v = int(scale[-1])

			if scale_v > 0:
				scale_v -= 1

			return "HU" + str(hue_v) + ",SH" + str(shape_v) + ",SC" + str(scale_v)

		start = cfg.data.class_names[start_idx]
		target = table[start]
		target_idx = cfg.data.class_names.index(target)
		return target_idx


	def compute_shape_target(start_idx):
		def table(start):
			hue, shape, scale = start.split(",")
			hue_v   = int(hue[-1])
			shape_v = int(shape[-1])
			scale_v = int(scale[-1])

			if shape_v < 4:
				shape_v += 1
			else:
				shape_v = 0

			return "HU" + str(hue_v) + ",SH" + str(shape_v) + ",SC" + str(scale_v)

		start = cfg.data.class_names[start_idx]
		target = table[start]
		target_idx = cfg.data.class_names.index(target)
		return target_idx

	if fix is not None:
		assert type(fix) == int
		more_hue = fix
		more_hue_sample = prior.gaussians[fix].rsample()
		less_hue = fix
		less_hue_sample = prior.gaussians[fix].rsample()
		bigger = fix
		bigger_sample = prior.gaussians[fix].rsample()
		smaller = fix
		smaller_sample = prior.gaussians[fix].rsample()
		shape = fix
		shape_sample = prior.gaussians[fix].rsample()
	else:
		more_hue = np.random.choice(digits_node_id)
		more_hue_sample = prior.gaussians[more_hue].rsample()
		less_hue = np.random.choice(digits_node_id)
		less_hue_sample = prior.gaussians[less_hue].rsample()
		bigger = np.random.choice(digits_node_id)
		bigger_sample = prior.gaussians[bigger].rsample()
		smaller = np.random.choice(digits_node_id)
		smaller_sample = prior.gaussians[smaller].rsample()
		shape = np.random.choice(digits_node_id)
		shape_sample = prior.gaussians[shape].rsample()

	more_hue_target = compute_more_hue_target(more_hue)
	more_hue_target_sample = prior.gaussians[more_hue_target].rsample()

	less_hue_target = compute_less_hue_target(less_hue)
	less_hue_target_sample = prior.gaussians[less_hue_target].rsample()

	bigger_target = compute_bigger_target(bigger)
	bigger_target_sample = prior.gaussians[bigger_target].rsample()

	smaller_target = compute_smaller_target(smaller)
	smaller_target_sample = prior.gaussians[smaller_target].rsample()

	shape_target = compute_shape_target(shape)
	shape_target_sample = prior.gaussians[shape_target].rsample()

	more_hue_data = Shapes3D_RelationalData(
		label=torch.tensor([more_hue], dtype=torch.long),
		sample=more_hue_sample,
		target_sample=more_hue_target_sample,
		target_label=torch.tensor([more_hue_target], dtype=torch.long)
	)
	less_hue_data = Shapes3D_RelationalData(
		label=torch.tensor([less_hue], dtype=torch.long),
		sample=less_hue_sample,
		target_sample=less_hue_target_sample,
		target_label=torch.tensor([less_hue_target], dtype=torch.long)
	)
	bigger_data = Shapes3D_RelationalData(
		label=torch.tensor([bigger], dtype=torch.long),
		sample=bigger_sample,
		target_sample=bigger_target_sample,
		target_label=torch.tensor([bigger_target], dtype=torch.long)
	)
	smaller_data = Shapes3D_RelationalData(
		label=torch.tensor([smaller], dtype=torch.long),
		sample=smaller_sample,
		target_sample=smaller_target_sample,
		target_label=torch.tensor([smaller_target], dtype=torch.long)
	)
	shape_data = Shapes3D_RelationalData(
		label=torch.tensor([shape], dtype=torch.long),
		sample=shape_sample,
		target_sample=shape_target_sample,
		target_label=torch.tensor([shape_target], dtype=torch.long)
	)
	return more_hue_data, less_hue_data, bigger_data, smaller_data, shape_data

def new_batch_of_Shapes3D_relations(cfg, batch_size, prior, digits_node_id, fix_more_hue=None):
	more_hue_relations = []
	less_hue_relations = []
	bigger_relations = []
	smaller_relations = []
	shape_relations = []

	for _ in range(batch_size):
		lr, rr, ur, dr, sr = _build_Shapes3D_relations(cfg, prior, digits_node_id, fix_more_hue)
		more_hue_relations.append(lr)
		less_hue_relations.append(rr)
		bigger_relations.append(ur)
		smaller_relations.append(dr)
		shape_relations.append(sr)

	batch = Shapes3D_RelationalBatch(cfg, more_hue_relations, less_hue_relations, bigger_relations, smaller_relations, shape_relations)
	return batch


class Shapes3D_RelationalData():
	def __init__(self, label, sample, target_label, target_sample):
		super(Shapes3D_RelationalData, self).__init__()
		self.label = label
		self.sample = sample
		self.target_sample = target_sample
		self.target_label = target_label

class Shapes3D_RelationalBatch():
	def __init__(self, cfg, more_hue_relations, less_hue_relations, bigger_relations, smaller_relations, shape_relations):
		super(Shapes3D_RelationalBatch, self).__init__()
		self.z_dim = cfg.model.z_dim

		self.more_hue_label = []
		self.more_hue_sample = []
		self.more_hue_target_label = []
		self.more_hue_target_sample = []

		for i, data in enumerate(more_hue_relations):
			self.more_hue_label.append(data.label)
			self.more_hue_sample.append(data.sample)
			self.more_hue_target_label.append(data.target_label)
			self.more_hue_target_sample.append(data.target_sample)

		self.more_hue_label = torch.cat(self.more_hue_label, dim=0)
		self.more_hue_sample = torch.vstack(self.more_hue_sample)
		self.more_hue_target_label = torch.cat(self.more_hue_target_label, dim=0)
		self.more_hue_target_sample = torch.vstack(self.more_hue_target_sample)

		self.less_hue_label = []
		self.less_hue_sample = []
		self.less_hue_target_label = []
		self.less_hue_target_sample = []

		for i, data in enumerate(less_hue_relations):
			self.less_hue_label.append(data.label)
			self.less_hue_sample.append(data.sample)
			self.less_hue_target_label.append(data.target_label)
			self.less_hue_target_sample.append(data.target_sample)

		self.less_hue_label = torch.cat(self.less_hue_label, dim=0)
		self.less_hue_sample = torch.vstack(self.less_hue_sample)
		self.less_hue_target_label = torch.cat(self.less_hue_target_label, dim=0)
		self.less_hue_target_sample = torch.vstack(self.less_hue_target_sample)

		self.bigger_label = []
		self.bigger_sample = []
		self.bigger_target_label = []
		self.bigger_target_sample = []

		for i, data in enumerate(bigger_relations):
			self.bigger_label.append(data.label)
			self.bigger_sample.append(data.sample)
			self.bigger_target_label.append(data.target_label)
			self.bigger_target_sample.append(data.target_sample)

		self.bigger_label = torch.cat(self.bigger_label, dim=0)
		self.bigger_sample = torch.vstack(self.bigger_sample)
		self.bigger_target_label = torch.cat(self.bigger_target_label, dim=0)
		self.bigger_target_sample = torch.vstack(self.bigger_target_sample)

		self.smaller_label = []
		self.smaller_sample = []
		self.smaller_target_label = []
		self.smaller_target_sample = []

		for i, data in enumerate(smaller_relations):
			self.smaller_label.append(data.label)
			self.smaller_sample.append(data.sample)
			self.smaller_target_label.append(data.target_label)
			self.smaller_target_sample.append(data.target_sample)

		self.smaller_label = torch.cat(self.smaller_label, dim=0)
		self.smaller_sample = torch.vstack(self.smaller_sample)
		self.smaller_target_label = torch.cat(self.smaller_target_label, dim=0)
		self.smaller_target_sample = torch.vstack(self.smaller_target_sample)

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
		self.more_hue_label = self.more_hue_label.to(device)
		self.more_hue_sample = self.more_hue_sample.to(device)
		self.more_hue_target_label = self.more_hue_target_label.to(device)
		self.more_hue_target_sample = self.more_hue_target_sample.to(device)

		self.less_hue_label = self.less_hue_label.to(device)
		self.less_hue_sample = self.less_hue_sample.to(device)
		self.less_hue_target_label = self.less_hue_target_label.to(device)
		self.less_hue_target_sample = self.less_hue_target_sample.to(device)

		self.bigger_label = self.bigger_label.to(device)
		self.bigger_sample = self.bigger_sample.to(device)
		self.bigger_target_label = self.bigger_target_label.to(device)
		self.bigger_target_sample = self.bigger_target_sample.to(device)

		self.smaller_label = self.smaller_label.to(device)
		self.smaller_sample = self.smaller_sample.to(device)
		self.smaller_target_label = self.smaller_target_label.to(device)
		self.smaller_target_sample = self.smaller_target_sample.to(device)

		self.shape_label = self.shape_label.to(device)
		self.shape_sample = self.shape_sample.to(device)
		self.shape_target_label = self.shape_target_label.to(device)
		self.shape_target_sample = self.shape_target_sample.to(device)

		return self
