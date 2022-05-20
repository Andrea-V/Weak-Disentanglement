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

class HWF(Dataset):
	def __init__(self, cfg, dataset_path):
		super(HWF, self).__init__()
		#data_transform = transforms.Compose([
		#	transforms.ToTensor()
		#])
		raw_data_path = os.path.join( dataset_path, "Handwritten_Math_Symbols")
		dir, subdirs, files = next(os.walk(raw_data_path))
		self.cfg = cfg
		self.data = []
		self.classes_idxs = {
			"0": 0,
			"1": 1,
			"2": 2,
			"3": 3,
			"4": 4,
			"5": 5,
			"6": 6,
			"7": 7,
			"8": 8,
			"9": 9,
			"+": 10,
			"-": 11,
			"times": 12#,
			#"div": 13
		}

		if not os.path.exists(os.path.join(dataset_path, "processed")):
			for subdir in subdirs:
				dir, subdirs, files = next(os.walk(os.path.join(raw_data_path, subdir)))
				class_name = dir.split("/")[-1]
				print("- Preprocessing class:", class_name)

				files = list(filter(lambda f: f.startswith(class_name), files))

				for file in tqdm.tqdm(files):
					image = Image.open(os.path.join(dir, file))
					image = torchvision.transforms.functional.pil_to_tensor(image).float()
					image /= 255.0
					class_idx = self.classes_idxs[class_name]
					self.data.append((image, class_idx, class_name))

			os.makedirs(os.path.join(dataset_path, "processed"))
			torch.save(self.data, os.path.join(dataset_path, "processed", "data.pt"))
		else:
			self.data = torch.load(os.path.join(dataset_path, "processed", "data.pt"))

		random.shuffle(self.data)

	def from_label_to_one_hot(self, label):
		res = np.array([0] * self.cfg.data.n_classes)
		res[self.classes_idxs[label]] = 1
		return res

	def __len__(self):
		return len(self.data)

	# the item MUST NOT have batch dimension
	def __getitem__(self, idx):

		y = self.data[idx][1]
		x = self.data[idx][0]
		label = self.data[idx][2]

		return x, y, label


def load_hwf(cfg, path, batch_size):
	def get_weights(label):
		weights = {
			"0": 1/2225, #1/6914
			"1": 1/8647, #1/26520
			"2": 1/8573, #1/26141
			"3": 1/3509, #1/10909
			"4": 1/2335, #1/7396
			"5": 1/1143, #1/3545
			"6": 1/990, #1/3118
			"7": 1/923, #1/2909
			"8": 1/965, #1/3068
			"9": 1/997, #1/3737
			"+": 1/8208, #1/25112
			"-": 1/11251, #1/33997
			"times": 1/908 #, #1/3251
			#"div":  1/217 #1/868
		}
		return weights[label]

	hwf = HWF(cfg, path)
	idxs = list(range(len(hwf)))

	labels  = np.array(hwf.data).T[2]
	weights = list(map(get_weights, labels))
     
    ten_percent = int(0.1 * len(idxs))
     
	# split, train, validation and test set
	tr_idxs    = idxs[:-2*ten_precent]
	tr_weights = weights[:-2*ten_precent]
	vl_idxs    = idxs[-2*ten_precent:-ten_precent]
	vl_weights = weights[-2*ten_precent:-ten_precent]
	ts_idxs    = idxs[-ten_precent:]
	ts_weights = weights[-ten_precent:]

	tr_sampler = UnbalancedDatasetSampler(tr_idxs, tr_weights, num_samples=len(tr_idxs), replacement=True)
	vl_sampler = UnbalancedDatasetSampler(vl_idxs, vl_weights, num_samples=len(vl_idxs), replacement=True)
	ts_sampler = UnbalancedDatasetSampler(ts_idxs, ts_weights, num_samples=len(ts_idxs), replacement=True)

	tr_loader = torch.utils.data.DataLoader(
		hwf,
		batch_size=batch_size,
		sampler=tr_sampler,
		num_workers=cfg.data.n_workers_tr,
		drop_last=True
	)

	vl_loader = torch.utils.data.DataLoader(
		hwf,
		batch_size=batch_size,
		sampler=vl_sampler,
		num_workers=cfg.data.n_workers_vl,
		drop_last=True
	)

	ts_loader = torch.utils.data.DataLoader(
		hwf,
		batch_size=batch_size,
		sampler=ts_sampler,
		num_workers=cfg.data.n_workers_ts,
		drop_last=True
	)

	return tr_loader, vl_loader, ts_loader




def _build_hwf_relations(cfg, prior, digits_node_id, fix_left=None):
	if fix_left is not None:
		assert type(fix_left) == int
		left_add = fix_left
		left_add_sample = prior.gaussians[fix_left].rsample()
		left_sub = fix_left
		left_sub_sample = prior.gaussians[fix_left].rsample()
		left_mul = fix_left
		left_mul_sample = prior.gaussians[fix_left].rsample()
	else:
		left_add = np.random.choice(digits_node_id)
		left_add_sample = prior.gaussians[left_add].rsample()
		left_sub = np.random.choice(digits_node_id)
		left_sub_sample = prior.gaussians[left_sub].rsample()
		left_mul = np.random.choice(digits_node_id)
		left_mul_sample = prior.gaussians[left_mul].rsample()

	right_add = np.random.choice(digits_node_id)
	right_add_sample = prior.gaussians[right_add].rsample()
	right_sub = np.random.choice(digits_node_id)
	right_sub_sample = prior.gaussians[right_sub].rsample()
	right_mul = np.random.choice(digits_node_id)
	right_mul_sample = prior.gaussians[right_mul].rsample()

	add_relation = prior.gaussians[10].rsample()
	add_target = (left_add + right_add) % 10
	add_target_sample = prior.gaussians[add_target].rsample()

	sub_relation = prior.gaussians[11].rsample()
	sub_target = (left_sub - right_sub) % 10
	sub_target_sample = prior.gaussians[sub_target].rsample()

	mul_relation = prior.gaussians[12].rsample()
	mul_target = (left_mul * right_mul) % 10
	mul_target_sample = prior.gaussians[mul_target].rsample()

	add_data = HWF_RelationalData(
		arg1_label=torch.tensor([left_add], dtype=torch.long),
		arg2_label=torch.tensor([right_add], dtype=torch.long),
		arg1_sample=left_add_sample,
		arg2_sample=right_add_sample,
		relation=add_relation,
		target_sample=add_target_sample,
		target_label=torch.tensor([add_target], dtype=torch.long)
	)
	sub_data = HWF_RelationalData(
		arg1_label=torch.tensor([left_sub], dtype=torch.long),
		arg2_label=torch.tensor([right_sub], dtype=torch.long),
		arg1_sample=left_sub_sample,
		arg2_sample=right_sub_sample,
		relation=sub_relation,
		target_sample=sub_target_sample,
		target_label=torch.tensor([sub_target], dtype=torch.long)
	)
	mul_data = HWF_RelationalData(
		arg1_label=torch.tensor([left_mul], dtype=torch.long),
		arg2_label=torch.tensor([right_mul], dtype=torch.long),
		arg1_sample=left_mul_sample,
		arg2_sample=right_mul_sample,
		relation=mul_relation,
		target_sample=mul_target_sample,
		target_label=torch.tensor([mul_target], dtype=torch.long)
	)
	return add_data, sub_data, mul_data

def new_batch_of_hwf_relations(cfg, batch_size, prior, digits_node_id, fix_left=None):
	#graph_list = [ _build_hwf_relation_graph(cfg, prior) for _ in range(batch_size)]

	add_relations = []
	sub_relations = []
	mul_relations = []

	for _ in range(batch_size):
		ar, sr, mr = _build_hwf_relations(cfg, prior, digits_node_id, fix_left)
		add_relations.append(ar)
		sub_relations.append(sr)
		mul_relations.append(mr)

	batch = HWF_RelationalBatch(cfg, graph_list=None, add_list=add_relations, sub_list=sub_relations, mul_list=mul_relations, batch_size=batch_size)
	return batch


class HWF_RelationalData():
	def __init__(self, arg1_label, arg2_label, arg1_sample, arg2_sample, relation, target_sample, target_label):
		super(HWF_RelationalData, self).__init__()
		self.arg1_label = arg1_label
		self.arg1_sample = arg1_sample
		self.arg2_label = arg2_label
		self.arg2_sample = arg2_sample
		self.relation = relation
		self.target_sample = target_sample
		self.target_label=target_label


class HWF_RelationalBatch():
	def __init__(self, cfg, graph_list, add_list, sub_list, mul_list, batch_size):
		super(HWF_RelationalBatch, self).__init__()
		self.batch_size = batch_size
		self.z_dim = cfg.model.z_dim

		#self.graph_batch = Batch().from_data_list(graph_list)

		add_left_label_batch = []
		add_left_sample_batch = []
		add_right_label_batch = []
		add_right_sample_batch = []
		add_relation_batch = []
		add_target_sample_batch = []
		add_target_label_batch = []

		for i, rel_data in enumerate(add_list):
			add_left_label_batch.append(rel_data.arg1_label)
			add_left_sample_batch.append(rel_data.arg1_sample)
			add_right_label_batch.append(rel_data.arg2_label)
			add_right_sample_batch.append(rel_data.arg2_sample)
			add_relation_batch.append(rel_data.relation)
			add_target_label_batch.append(rel_data.target_label)
			add_target_sample_batch.append(rel_data.target_sample)

		self.add_left_label = torch.vstack(add_left_label_batch)
		self.add_left_sample = torch.vstack(add_left_sample_batch)
		self.add_right_label = torch.vstack(add_right_label_batch)
		self.add_right_sample = torch.vstack(add_right_sample_batch)
		self.add_relation = torch.vstack(add_relation_batch)
		self.add_target_label = torch.vstack(add_target_label_batch)
		self.add_target_sample = torch.vstack(add_target_sample_batch)

		sub_left_label_batch = []
		sub_left_sample_batch = []
		sub_right_label_batch = []
		sub_right_sample_batch = []
		sub_relation_batch = []
		sub_target_sample_batch = []
		sub_target_label_batch = []

		for i, rel_data in enumerate(sub_list):
			sub_left_label_batch.append(rel_data.arg1_label)
			sub_left_sample_batch.append(rel_data.arg1_sample)
			sub_right_label_batch.append(rel_data.arg2_label)
			sub_right_sample_batch.append(rel_data.arg2_sample)
			sub_relation_batch.append(rel_data.relation)
			sub_target_label_batch.append(rel_data.target_label)
			sub_target_sample_batch.append(rel_data.target_sample)

		self.sub_left_label = torch.vstack(sub_left_label_batch)
		self.sub_left_sample = torch.vstack(sub_left_sample_batch)
		self.sub_right_label = torch.vstack(sub_right_label_batch)
		self.sub_right_sample = torch.vstack(sub_right_sample_batch)
		self.sub_relation = torch.vstack(sub_relation_batch)
		self.sub_target_label = torch.vstack(sub_target_label_batch)
		self.sub_target_sample = torch.vstack(sub_target_sample_batch)

		mul_left_label_batch = []
		mul_left_sample_batch = []
		mul_right_label_batch = []
		mul_right_sample_batch = []
		mul_relation_batch = []
		mul_target_sample_batch = []
		mul_target_label_batch = []

		for i, rel_data in enumerate(mul_list):
			mul_left_label_batch.append(rel_data.arg1_label)
			mul_left_sample_batch.append(rel_data.arg1_sample)
			mul_right_label_batch.append(rel_data.arg2_label)
			mul_right_sample_batch.append(rel_data.arg2_sample)
			mul_relation_batch.append(rel_data.relation)
			mul_target_label_batch.append(rel_data.target_label)
			mul_target_sample_batch.append(rel_data.target_sample)

		self.mul_left_label = torch.vstack(mul_left_label_batch)
		self.mul_left_sample = torch.vstack(mul_left_sample_batch)
		self.mul_right_label = torch.vstack(mul_right_label_batch)
		self.mul_right_sample = torch.vstack(mul_right_sample_batch)
		self.mul_relation = torch.vstack(mul_relation_batch)
		self.mul_target_label = torch.vstack(mul_target_label_batch)
		self.mul_target_sample = torch.vstack(mul_target_sample_batch)

	def to(self, device):
		self.add_left_label = self.add_left_label.to(device)
		self.add_left_sample = self.add_left_sample.to(device)
		self.add_right_label = self.add_right_label.to(device)
		self.add_right_sample = self.add_right_sample.to(device)
		self.add_relation = self.add_relation.to(device)
		self.add_target_label = self.add_target_label.to(device)
		self.add_target_sample = self.add_target_sample.to(device)

		self.sub_left_label = self.sub_left_label.to(device)
		self.sub_left_sample = self.sub_left_sample.to(device)
		self.sub_right_label = self.sub_right_label.to(device)
		self.sub_right_sample = self.sub_right_sample.to(device)
		self.sub_relation = self.sub_relation.to(device)
		self.sub_target_label = self.sub_target_label.to(device)
		self.sub_target_sample = self.sub_target_sample.to(device)

		self.mul_left_label = self.mul_left_label.to(device)
		self.mul_left_sample = self.mul_left_sample.to(device)
		self.mul_right_label = self.mul_right_label.to(device)
		self.mul_right_sample = self.mul_right_sample.to(device)
		self.mul_relation = self.mul_relation.to(device)
		self.mul_target_label = self.mul_target_label.to(device)
		self.mul_target_sample = self.mul_target_sample.to(device)

		return self
		
