import torch
import os
import json
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.autograd
import pickle
import pprint as pp
import prior
from sklearn.decomposition import PCA
import numpy as np
from itertools import tee
import random
import sys
import torch.nn.functional as F
import dataset as mod_dset

# moved to the top level bc otherwise pickle won't work
def sigmoid(x):
	return torch.sigmoid(x)
def tanh(x):
	return torch.tanh(x)
def relu(x):
	return torch.relu(x)
def linear(x):
	return x

def store_model(cfg, model, path, model_name):
	store_path = os.path.join(path, cfg.name)
	if not os.path.exists(store_path):
		os.makedirs(store_path)

	torch.save(model, os.path.join(store_path, model_name + ".pt"))
	
	with open(os.path.join(store_path, model_name + ".cfg"), "w") as fp:
		pp.pprint(cfg, stream=fp)

	with open(os.path.join(store_path, model_name + ".model"), "w") as fp:
		pp.pprint(model, stream=fp)

	with open(os.path.join(store_path, model_name + ".cfg.pkl"), "wb") as fp:
		pickle.dump(cfg, fp)

def store_adaptive_prior(cfg, model, path, model_name):
	store_path = os.path.join(path, cfg.name, "embeddings")
	if not os.path.exists(store_path):
		os.makedirs(store_path)

	#assert type(model.prior) == prior.AdaptiveMixtureOfGaussianPrior

	with open(os.path.join(store_path, model_name + ".pkl"), "wb") as fp:
		model.prior.to("cpu")
		pickle.dump(model.prior, fp)
		model.prior.to(cfg.device)


def store_training_data(cfg, plot_data, path, plot_data_name):
	plot_data_path = os.path.join(path, cfg.name, "data")
	if not os.path.exists(plot_data_path):
		os.makedirs(plot_data_path)

	with open(os.path.join(plot_data_path, plot_data_name + ".json"), "w") as fp:
		json.dump(plot_data, fp)

def plot_training_data(cfg, data, path, name):
	plot_path = os.path.join(path, cfg.name)
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	tr_data = data[data["dataset"] == "training"]
	vl_data = data[data["dataset"] == "validation"]

	tr_data_ae  = tr_data[tr_data["type"].isin(["recon", "prior", "gaussian", "categorical", "relational", "semi-sup"])]
	tr_data_gan = tr_data[tr_data["type"].isin(["adv_gen", "adv_disc"])]
	tr_data_scores = tr_data[tr_data["type"].isin(["gen_score", "prior_score", "inter_score"])]#, "grad_penalty"])]

	vl_data_ae  = vl_data[vl_data["type"].isin(["recon", "prior", "gaussian", "categorical", "relational", "semi-sup"])]
	vl_data_gan = vl_data[vl_data["type"].isin(["adv_gen", "adv_disc"])]
	vl_data_scores = vl_data[vl_data["type"].isin(["gen_score", "prior_score", "inter_score"])]#, "grad_penalty"])]

	sns.set_style("whitegrid")
	plt.figure(figsize=(20, 10))
	plt.subplot(3, 2, 1)
	ax = sns.lineplot(x="epoch", y="loss", hue="type", data=tr_data_ae)
	plt.ylim((-0.01, 1.01))
	ax.set_title("Training - AE")
	plt.subplot(3, 2, 3)
	ax = sns.lineplot(x="epoch", y="loss", hue="type", data=tr_data_gan)
	ax.set_title("Training - GAN")

	plt.subplot(3, 2, 5)
	ax = sns.lineplot(x="epoch", y="loss", hue="type", data=tr_data_scores)
	ax.set_title("Training - Scores")

	plt.subplot(3, 2, 2)
	ax = sns.lineplot(x="epoch", y="loss", hue="type", data=vl_data_ae)
	plt.ylim((-0.01, 1.01))
	ax.set_title("Validation - AE")
	plt.subplot(3, 2, 4)
	ax = sns.lineplot(x="epoch", y="loss", hue="type", data=vl_data_gan)
	ax.set_title("Validation - GAN")
	plt.subplot(3, 2, 6)
	ax = sns.lineplot(x="epoch", y="loss", hue="type", data=vl_data_scores)
	ax.set_title("Validation - Scores")

	plt.savefig(os.path.join(plot_path, name) + ".png")
	plt.close()


def plot_latent_space_abs(cfg, model, tr_set, vl_set, path, name):
	plot_path = os.path.join(path, cfg.name, "latent_space")
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)
	dataset = {"TR": tr_set,"VL": vl_set}
	prior_z = {"TR": {"z": [], "z1": [],"z2": [],"y": [],"label": []},"VL":{"z": [], "z1": [],"z2": [],"y": [],"label": []}}
	posterior_z = {"TR": {"z": [], "z1": [],"z2": [],"y": [],"label": []},"VL":{"z": [], "z1": [],"z2": [],"y": [],"label": []}}

	class_names = cfg.data.class_names

	with torch.no_grad():
		for ds in ["TR", "VL"]:
			for i, (x, y, label) in enumerate(dataset[ds]):

				if i > cfg.log.max_batch_to_plot:
					break

				if cfg.current_dataset == "HWF":
					x = x.view((-1, 1, 45, 45)).float().to(cfg.device)
				else:
					raise ValueError("Unknown dataset.")

				y_1hot = F.one_hot(y, num_classes=cfg.data.n_classes).to(cfg.device)

				#y_prior = model.prior.categorical.sample((cfg.data.batch_size,)).to(cfg.device)

				z_prior = model.prior.sample(y_1hot)
				z_post = model.encoder(x) #, y_1hot)

				#y_prior = y_prior.argmax(dim=-1).cpu().numpy()
				#y_post = y_post.argmax(dim=-1).cpu().numpy()
				
				z_prior = z_prior.cpu().numpy() 
				z_post = z_post.cpu().numpy()
				
				y = y.cpu().numpy()
				#y = [class_names[label_i] for label_i in label]
				#y_prior = [class_names[label_i] for label_i in y_prior]

				prior_z[ds]["z"].extend(list(z_prior))
				prior_z[ds]["z1"].extend(list(z_prior.T[0]))
				prior_z[ds]["z2"].extend(list(z_prior.T[1]))
				prior_z[ds]["label"].extend(list(label))
				prior_z[ds]["y"].extend(list(y))
				
				posterior_z[ds]["z"].extend(list(z_post))
				posterior_z[ds]["z1"].extend(list(z_post.T[0]))
				posterior_z[ds]["z2"].extend(list(z_post.T[1]))
				posterior_z[ds]["label"].extend(label)
				posterior_z[ds]["y"].extend(list(y))



		# if z_dim > 2, fare PCA e plottare i primi due PC
		if cfg.model.z_dim > 2:
			pca_prior     = PCA(n_components=2).fit(prior_z[ds]["z"])
			pca_posterior = PCA(n_components=2).fit(posterior_z[ds]["z"])

			transformed_prior = np.array(pca_prior.transform(prior_z[ds]["z"]))
			transformed_posterior = np.array(pca_posterior.transform(posterior_z[ds]["z"]))

			prior_z[ds]["z1"] = list(transformed_prior.T[0])
			prior_z[ds]["z2"] = list(transformed_prior.T[1])

			posterior_z[ds]["z1"] = list(transformed_posterior.T[0])
			posterior_z[ds]["z2"] = list(transformed_posterior.T[1])

		del prior_z[ds]["z"]
		del posterior_z[ds]["z"]

	prior_z_tr     = pd.DataFrame.from_dict(prior_z["TR"])
	posterior_z_tr = pd.DataFrame.from_dict(posterior_z["TR"])

	prior_z_tr.sort_values(by=["label"], inplace=True)
	posterior_z_tr.sort_values(by=["label"], inplace=True)

	#palette_prior = sns.color_palette(n_colors = cfg.data.n_classes)
	#palette_posterior = sns.color_palette(n_colors = model.prior.n_classes)
	sns.set_style("whitegrid")
	plt.figure(figsize=(20, 10))
	try:
		ax1 = plt.subplot(2, 2, 1)
		sns.kdeplot(x=prior_z_tr["z1"], y=prior_z_tr["z2"], ax=ax1, shade=False)
		ax1.set_title("TR: Latent space - prior")
	except:
		pass

	ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
	sns.scatterplot(x="z1", y="z2", hue="label", data=prior_z_tr, ax=ax2, legend=False)#, palette=palette_prior)
	ax2.set_title("TR: Latent space - prior")

	try:
		ax3 = plt.subplot(2, 2, 3)
		sns.kdeplot(x=posterior_z_tr["z1"], y=posterior_z_tr["z2"], ax=ax3, shade=False)
		ax3.set_title("TR: Latent space - posterior")
	except:
		pass

	ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax3)
	sns.scatterplot(x="z1", y="z2", hue="label", data=posterior_z_tr, ax=ax4, legend=False)#, palette=palette_posterior)
	ax4.set_title("TR: Latent space - posterior")

	# ax5 = plt.subplot(2, 4, 5)
	# ax5 = sns.scatterplot(x="x", y="y", data=centroids_df, ax=ax5)
	# #ax5 = ax5.facet_axis(0,0)

	# for m, s in zip(mu_prior, sigma_prior):
	# 	#print(m, s)
	# 	elps = Ellipse(m, s[0]/2, s[1]/2,edgecolor="b", facecolor='none')
	# 	ax5.add_artist(elps)

	# ax5.set_xlim(mu_prior.T[0].min() - (sigma_prior[mu_prior.T[0].argmin()][0] / 2), mu_prior.T[0].max() + (sigma_prior[mu_prior.T[0].argmax()][0] / 2))
	# ax5.set_ylim(mu_prior.T[1].min() - (sigma_prior[mu_prior.T[1].argmin()][1] / 2), mu_prior.T[1].max() + (sigma_prior[mu_prior.T[1].argmax()][1] / 2))
	# ax5.set_title("Mixture Components - prior")
	
	# ax6 = plt.subplot(2, 4, 6)
	# sns.histplot(data=prior_z_tr["y"], bins=cfg.data.n_classes, ax=ax6)
	# ax6.set_title("Y latents - prior")

	# ax7 = plt.subplot(2, 4, 5)
	# ax7 = sns.scatterplot(x="x", y="y", data=centroids_df, ax=ax5)
	# #ax5 = ax5.facet_axis(0,0)

	# for m, s in zip(mu_prior, sigma_prior):
	# 	#print(m, s)
	# 	elps = Ellipse(m, s[0]/2, s[1]/2,edgecolor="b", facecolor='none')
	# 	ax7.add_artist(elps)

	# ax7.set_xlim(mu_prior.T[0].min() - (sigma_prior[mu_prior.T[0].argmin()][0] / 2), mu_prior.T[0].max() + (sigma_prior[mu_prior.T[0].argmax()][0] / 2))
	# ax7.set_ylim(mu_prior.T[1].min() - (sigma_prior[mu_prior.T[1].argmin()][1] / 2), mu_prior.T[1].max() + (sigma_prior[mu_prior.T[1].argmax()][1] / 2))
	# ax7.set_title("Mixture Components - posterior")
	
	# ax8 = plt.subplot(2, 4, 8)
	# sns.histplot(data=posterior_z_tr["y"], bins=cfg.data.n_classes, ax=ax8)
	# ax8.set_title("Y latents - posterior")

	plt.savefig(os.path.join(plot_path, name) + ".png")
	plt.close()

def plot_adaptive_prior(cfg, model, n_samples, path, name):
	plot_path = os.path.join(path, cfg.name, "adaptive_prior")
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	# cifar images are normalized, denormalize them before plotting
	#denormalizer = Denormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))

	model_prior = model.prior

	# for gaussian in model_prior.gaussians:
	# 	print(gaussian)
	# 	print(gaussian.loc)
	# 	print(gaussian.loc.device)
	# 	print(gaussian._unbroadcasted_scale_tril)
	# 	print(gaussian._unbroadcasted_scale_tril.device)
	#
	# sys.exit(0)
	#print(type(model_prior))

	#adaptive_gaussmix_prior = type(model_prior) == prior.AdaptiveMixtureOfGaussianPrior
	#flower_prior = type(model_prior) == prior.Flower
	#assert adaptive_gaussmix_prior or flower_prior

	x = []
	x_centroids = []
	for gaussian in model_prior.gaussians:

		z_i = gaussian.rsample((n_samples,)).to(cfg.device)

		with torch.no_grad():
			c_i = model.decoder(gaussian.loc)
			c_i = torch.sigmoid(c_i)

			x_i = model.decoder(z_i)
			x_i = torch.sigmoid(x_i)

			if cfg.current_dataset == "HWF":
				x_i = x_i.view((n_samples * 45, 45))
				c_i = c_i.view((45, 45))
			else:
				raise ValueError("Unknown dataset")
			#x_i = x_i.cpu().numpy()
			# print("x_i:", x_i.shape)
			# print("c_i:", c_i.shape)
			
			x_centroids.append(c_i)
			x.append(x_i)


	x = torch.cat(x, dim=-1)
	x_centroids = torch.cat(x_centroids, dim=-1)

	#print("x_shape:", x.shape)
	#print("c shape:", x_centroids.shape)

	x = x.cpu()
	x_centroids = x_centroids.cpu()

	plt.figure(figsize=(model.prior.n_classes, n_samples))

	sns.heatmap(x)

	plt.savefig(os.path.join(plot_path, name + "samples") + ".png")
	plt.close()
	
	plt.figure(figsize=(model.prior.n_classes, 2.5))


	sns.heatmap(x_centroids)

	plt.savefig(os.path.join(plot_path, name + "centroids") + ".png")
	plt.close()

def plot_gaussian_prior(cfg, model, n_samples, path, name):
	plot_path = os.path.join(path, cfg.name, "gaussian_prior")
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	# cifar images are normalized, denormalize them before plotting
	#denormalizer = Denormalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))

	model_prior = model.prior
	#print(type(model_prior))

	assert type(model_prior) == prior.Gaussian

	x = []

	for _ in range(10):
		z_sample = model_prior(n_samples)
		z_sample = z_sample.to(cfg.device)

		with torch.no_grad():
			x_i = model.decoder(z_sample)
			x_i = torch.sigmoid(x_i)

			#print("x_i:", x_i.shape)

			if cfg.current_dataset == "HWF":
				x_i = x_i.view((n_samples * 45, 45))
			else:
				raise ValueError("Unknown dataset")
				#x_centroids.append(c_i)

		x.append(x_i)

	x = torch.cat(x, dim=-1)
	#x_centroids = torch.cat(x_centroids, dim=-1)

	#print("x_shape:", x.shape)
	#print("c shape:", x_centroids.shape)

	x = x.cpu()
	#x_centroids = x_centroids.cpu()

	plt.figure(figsize=(10, n_samples))

	sns.heatmap(x)

	plt.savefig(os.path.join(plot_path, name + "samples") + ".png")
	plt.close()

	#plt.figure(figsize=(model.prior.n_classes, 2.5))

	#if cfg.current_dataset == "CIFAR10":
	#	plt.imshow(x_centroids)
	#else:
	#	sns.heatmap(x_centroids)

	#plt.savefig(os.path.join(plot_path, name + "centroids") + ".png")
	#plt.close()


def store_accuracy_data(cfg, data, path, name):
	data_path = os.path.join(path, cfg.name, "data")
	if not os.path.exists(data_path):
		os.makedirs(data_path)

	with open(os.path.join(data_path, name + ".json"), "w") as fp:
		json.dump(data, fp)

def plot_accuracy_data(cfg, data, path, name):
	plot_path = os.path.join(path, cfg.name)
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	tr_data = data[data["dataset"] == "training"]
	vl_data = data[data["dataset"] == "validation"]

	#tr_data_accepted  = tr_data[tr_data["type"].isin(["recon", "prior", "gaussian", "categorical"])]
	#tr_data_accuracy = tr_data[tr_data["type"].isin(["adv_gen", "adv_disc"])]

	sns.set_style("whitegrid")
	plt.figure(figsize=(20, 10))
	plt.subplot(2, 2, 1)
	ax = sns.lineplot(x="epoch", y="accepted_ratio", hue="alpha", data=tr_data)
	ax.set_title("Training - accepted_ratio")
	plt.subplot(2, 2, 3)
	ax = sns.lineplot(x="epoch", y="accuracy", hue="alpha", data=tr_data)
	ax.set_title("Training - accuracy")

	plt.subplot(2, 2, 2)
	ax = sns.lineplot(x="epoch", y="accepted_ratio", hue="alpha", data=vl_data)
	ax.set_title("Validation - accepted_ratio")
	plt.subplot(2, 2, 4)
	ax = sns.lineplot(x="epoch", y="accuracy", hue="alpha", data=vl_data)
	ax.set_title("Validation - accuracy")

	plt.savefig(os.path.join(plot_path, name) + ".png")
	plt.close()

def compute_clustering_accuracy(cfg, model, dataset):
	results = {}
	for alpha in cfg.eval.alpha_thresholds:
		results[alpha] = {"accepted_ratio": None, "accuracy": None}

	# replicate iterable for each alpha level
	dataset_iters = tee(dataset, len(cfg.eval.alpha_thresholds))

	#print(dataset_iters, len(dataset_iters))
	#exit("SFSG")

	with torch.no_grad():
		for dataset_iter, alpha in zip(dataset_iters, cfg.eval.alpha_thresholds):
			#print("- alpha level:", alpha, "log:", np.log(alpha))
			y_preds = []
			ys = []

			for i, (x, y, label) in enumerate(dataset_iter):


				x = x.view((-1, 1, 45, 45)).float().to(cfg.device)

				y = y.to(cfg.device).float()

				z = model.encoder(x)
				y_pred = model.prior.classify(z, threshold=alpha)

				y_preds.append(y_pred)
				ys.append(y)

			y_true = torch.cat(ys, dim=0)
			y_pred = torch.cat(y_preds, dim=0)
			accepted_y_pred = y_pred[y_pred != -1]
			accepted_y_true = y_true[y_pred != -1]

			accepted_ratio = float(accepted_y_pred.shape[0]) / float(y_pred.shape[0])
			
			try:
				accuracy = float((accepted_y_pred == accepted_y_true).sum()) / float(accepted_y_true.shape[0])
			except ZeroDivisionError:
				accuracy = 0.0

			results[alpha]["accepted_ratio"] = accepted_ratio
			results[alpha]["accuracy"] = accuracy
	return results

def plot_relation(cfg, model, n_samples, path, name):
	relation_path = os.path.join(path, cfg.name, "relation")
	if not os.path.exists(relation_path):
		os.makedirs(relation_path)

	raise NotImplementedError("plot_relation")

def compute_relation_accuracy(cfg, model, entities_ids):
	results = {}
	for alpha in cfg.eval.alpha_thresholds:
		results[alpha] = {
			"accepted_ratio_add": None,
			"accuracy_add": None,
			"accepted_ratio_sub": None,
			"accuracy_sub": None,
			"accepted_ratio_mul": None,
			"accuracy_mul": None,
			"accepted_ratio_tot": None,
			"accuracy_tot": None
		}

	# replicate iterable for each alpha level
	dataset = (
		mod_dset.new_batch_of_hwf_relations(cfg, cfg.data.prior_batch_size, model.prior, entities_ids)
		for _ in range(10)
	)

	dataset_iters = tee(dataset, len(cfg.eval.alpha_thresholds))

	with torch.no_grad():
		for dataset_iter, alpha in zip(dataset_iters, cfg.eval.alpha_thresholds):
			y_pred_add = []
			y_pred_sub = []
			y_pred_mul = []

			y_true_add = []
			y_true_sub = []
			y_true_mul = []

			for i, batch in enumerate(dataset_iter):
				batch = batch.to(cfg.device)

				z_pred = model.prior(batch)

				z_pred_add = z_pred[0]
				z_pred_sub = z_pred[1]
				z_pred_mul = z_pred[2]

				y_pred_add.append(model.prior.classify(z_pred_add, threshold=alpha))
				y_pred_sub.append(model.prior.classify(z_pred_sub, threshold=alpha))
				y_pred_mul.append(model.prior.classify(z_pred_mul, threshold=alpha))

				y_true_add.append(batch.add_target_label.squeeze())
				y_true_sub.append(batch.sub_target_label.squeeze())
				y_true_mul.append(batch.mul_target_label.squeeze())

			y_true_add = torch.cat(y_true_add, dim=0)
			y_true_sub = torch.cat(y_true_sub, dim=0)
			y_true_mul = torch.cat(y_true_mul, dim=0)

			y_pred_add = torch.cat(y_pred_add, dim=0)
			y_pred_sub = torch.cat(y_pred_sub, dim=0)
			y_pred_mul = torch.cat(y_pred_mul, dim=0)

			accepted_y_pred_add = y_pred_add[y_pred_add != -1]
			accepted_y_pred_sub = y_pred_sub[y_pred_sub != -1]
			accepted_y_pred_mul = y_pred_mul[y_pred_mul != -1]

			accepted_y_true_add = y_true_add[y_pred_add != -1]
			accepted_y_true_sub = y_true_sub[y_pred_sub != -1]
			accepted_y_true_mul = y_true_mul[y_pred_mul != -1]

			accepted_ratio_add = float(accepted_y_pred_add.shape[0]) / float(y_pred_add.shape[0])
			accepted_ratio_sub = float(accepted_y_pred_sub.shape[0]) / float(y_pred_sub.shape[0])
			accepted_ratio_mul = float(accepted_y_pred_mul.shape[0]) / float(y_pred_mul.shape[0])
			accepted_ratio_tot = (accepted_ratio_add + accepted_ratio_sub + accepted_ratio_mul) / 3.

			try:
				accuracy_add = float((accepted_y_pred_add == accepted_y_true_add).sum()) / float(accepted_y_true_add.shape[0])
			except ZeroDivisionError:
				accuracy_add = 0.0

			try:
				accuracy_sub = float((accepted_y_pred_sub == accepted_y_true_sub).sum()) / float(accepted_y_true_sub.shape[0])
			except ZeroDivisionError:
				accuracy_sub = 0.0

			try:
				accuracy_mul = float((accepted_y_pred_mul == accepted_y_true_mul).sum()) / float(accepted_y_true_mul.shape[0])
			except ZeroDivisionError:
				accuracy_mul = 0.0

			accuracy_tot = (accuracy_add + accuracy_sub + accuracy_mul) / 3.

			results[alpha]["accepted_ratio_add"] = accepted_ratio_add
			results[alpha]["accuracy_add"] = accuracy_add
			results[alpha]["accepted_ratio_sub"] = accepted_ratio_sub
			results[alpha]["accuracy_sub"] = accuracy_sub
			results[alpha]["accepted_ratio_mul"] = accepted_ratio_mul
			results[alpha]["accuracy_mul"] = accuracy_mul
			results[alpha]["accepted_ratio_tot"] = accepted_ratio_tot
			results[alpha]["accuracy_tot"] = accuracy_tot

	return results

# entities_ids: set of fixed left-term entitities to compute the accuracy
# other entities_ids: other entities to sample for right term
def compute_relation_accuracy_foreach_entity(cfg, model, fixed_entities_ids, other_entities_ids):
	results = {}
	for entity in fixed_entities_ids:
		results[entity] = {}

		for alpha in cfg.eval.alpha_thresholds:
			results[entity][alpha] = {
				"accepted_ratio_add": None,
				"accuracy_add": None,
				"accepted_ratio_sub": None,
				"accuracy_sub": None,
				"accepted_ratio_mul": None,
				"accuracy_mul": None,
				"accepted_ratio_tot": None,
				"accuracy_tot": None
			}

		# replicate iterable for each alpha level
		dataset = (
			mod_dset.new_batch_of_hwf_relations(cfg, 1024, model.prior, other_entities_ids, fix_left=entity)
			for _ in range(10)
		)

		dataset_iters = tee(dataset, len(cfg.eval.alpha_thresholds))

		with torch.no_grad():
			for dataset_iter, alpha in zip(dataset_iters, cfg.eval.alpha_thresholds):
				y_pred_add = []
				y_pred_sub = []
				y_pred_mul = []

				y_true_add = []
				y_true_sub = []
				y_true_mul = []

				for i, batch in enumerate(dataset_iter):
					batch = batch.to(cfg.device)

					z_pred = model.prior(batch)

					z_pred_add = z_pred[0]
					z_pred_sub = z_pred[1]
					z_pred_mul = z_pred[2]

					y_pred_add.append(model.prior.classify(z_pred_add, threshold=alpha))
					y_pred_sub.append(model.prior.classify(z_pred_sub, threshold=alpha))
					y_pred_mul.append(model.prior.classify(z_pred_mul, threshold=alpha))
					y_true_add.append(batch.add_target_label.squeeze())
					y_true_sub.append(batch.sub_target_label.squeeze())
					y_true_mul.append(batch.mul_target_label.squeeze())

				y_true_add = torch.cat(y_true_add, dim=0)
				y_true_sub = torch.cat(y_true_sub, dim=0)
				y_true_mul = torch.cat(y_true_mul, dim=0)

				y_pred_add = torch.cat(y_pred_add, dim=0)
				y_pred_sub = torch.cat(y_pred_sub, dim=0)
				y_pred_mul = torch.cat(y_pred_mul, dim=0)

				accepted_y_pred_add = y_pred_add[y_pred_add != -1]
				accepted_y_pred_sub = y_pred_sub[y_pred_sub != -1]
				accepted_y_pred_mul = y_pred_mul[y_pred_mul != -1]

				accepted_y_true_add = y_true_add[y_pred_add != -1]
				accepted_y_true_sub = y_true_sub[y_pred_sub != -1]
				accepted_y_true_mul = y_true_mul[y_pred_mul != -1]

				accepted_ratio_add = float(accepted_y_pred_add.shape[0]) / float(y_pred_add.shape[0])
				accepted_ratio_sub = float(accepted_y_pred_sub.shape[0]) / float(y_pred_sub.shape[0])
				accepted_ratio_mul = float(accepted_y_pred_mul.shape[0]) / float(y_pred_mul.shape[0])
				accepted_ratio_tot = (accepted_ratio_add + accepted_ratio_sub + accepted_ratio_mul) / 3.

				try:
					accuracy_add = float((accepted_y_pred_add == accepted_y_true_add).sum()) / float(accepted_y_true_add.shape[0])
				except ZeroDivisionError:
					accuracy_add = 0.0

				try:
					accuracy_sub = float((accepted_y_pred_sub == accepted_y_true_sub).sum()) / float(accepted_y_true_sub.shape[0])
				except ZeroDivisionError:
					accuracy_sub = 0.0

				try:
					accuracy_mul = float((accepted_y_pred_mul == accepted_y_true_mul).sum()) / float(accepted_y_true_mul.shape[0])
				except ZeroDivisionError:
					accuracy_mul = 0.0

				accuracy_tot = (accuracy_add + accuracy_sub + accuracy_mul) / 3.

				results[entity][alpha]["accepted_ratio_add"] = accepted_ratio_add
				results[entity][alpha]["accuracy_add"] = accuracy_add
				results[entity][alpha]["accepted_ratio_sub"] = accepted_ratio_sub
				results[entity][alpha]["accuracy_sub"] = accuracy_sub
				results[entity][alpha]["accepted_ratio_mul"] = accepted_ratio_mul
				results[entity][alpha]["accuracy_mul"] = accuracy_mul
				results[entity][alpha]["accepted_ratio_tot"] = accepted_ratio_tot
				results[entity][alpha]["accuracy_tot"] = accuracy_tot

	return results


def adapt_accuracy_to_dataframe(fold, epoch, accuracy, df_accuracy, dataset):

	for alpha_level in accuracy:
		df_accuracy = df_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio"],
				"accuracy": accuracy[alpha_level]["accuracy"],
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)

	return df_accuracy


def adapt_relation_accuracy_to_dataframe(fold, epoch, accuracy, df_rel_accuracy, dataset):
	for alpha_level in accuracy:
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_add"],
				"accuracy": accuracy[alpha_level]["accuracy_add"],
				"relation": "add",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_sub"],
				"accuracy": accuracy[alpha_level]["accuracy_sub"],
				"relation": "sub",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_mul"],
				"accuracy": accuracy[alpha_level]["accuracy_mul"],
				"relation": "mul",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_tot"],
				"accuracy": accuracy[alpha_level]["accuracy_tot"],
				"relation": "tot",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
	return df_rel_accuracy

def adapt_relation_accuracy_foreach_to_dataframe(fold, epoch, accuracy, df_rel_accuracy_foreach, dataset):
	for entity in accuracy:
		for alpha_level in accuracy[entity]:
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"entity": entity,
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_add"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_add"],
					"relation": "add",
					"alpha": alpha_level,
					"dataset": dataset
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"entity": entity,
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_sub"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_sub"],
					"relation": "sub",
					"alpha": alpha_level,
					"dataset": dataset
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"entity": entity,
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_mul"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_mul"],
					"relation": "mul",
					"alpha": alpha_level,
					"dataset": dataset
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"entity": entity,
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_tot"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_tot"],
					"relation": "tot",
					"alpha": alpha_level,
					"dataset": dataset
				},
				ignore_index=True
			)
	return df_rel_accuracy_foreach

def store_relation_accuracy_data(cfg, data, path, name):
	data_path = os.path.join(path, cfg.name, "data")
	if not os.path.exists(data_path):
		os.makedirs(data_path)

	with open(os.path.join(data_path, name + ".json"), "w") as fp:
		json.dump(data, fp)


def plot_relation_accuracy_data(cfg, data, path, name):
	plot_path = os.path.join(path, cfg.name, "relation_accuracy")
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	tr_data = data[data["dataset"] == "training"]
	vl_data = data[data["dataset"] == "validation"]

	tr_add_data = tr_data[tr_data["relation"] == "add"]
	tr_sub_data = tr_data[tr_data["relation"] == "sub"]
	tr_mul_data = tr_data[tr_data["relation"] == "mul"]
	tr_tot_data = tr_data[tr_data["relation"] == "tot"]

	vl_add_data = vl_data[vl_data["relation"] == "add"]
	vl_sub_data = vl_data[vl_data["relation"] == "sub"]
	vl_mul_data = vl_data[vl_data["relation"] == "mul"]
	vl_tot_data = vl_data[vl_data["relation"] == "tot"]

	data_dict = {
		"add_tr": tr_add_data,
		"sub_tr": tr_sub_data,
		"mul_tr": tr_mul_data,
		"add_vl": vl_add_data,
		"sub_vl": vl_sub_data,
		"mul_vl": vl_mul_data,
		"tot_tr": tr_tot_data,
		"tot_vl": vl_tot_data
	}

	for relation_name in ["add", "sub", "mul"]:
		sns.set_style("whitegrid")
		plt.figure(figsize=(20, 10))
		plt.subplot(2, 2, 1)
		ax = sns.lineplot(
			x="epoch",
			y="accepted_ratio",
			hue="alpha",
			data=data_dict[relation_name + "_tr"]
		)
		ax.set_title("Training - accepted_ratio - " + relation_name)
		plt.subplot(2, 2, 3)
		ax = sns.lineplot(
			x="epoch",
			y="accuracy",
			hue="alpha",
			style="relation",
			data=data_dict[relation_name + "_tr"]
		)
		ax.set_title("Training - accuracy - " + relation_name)

		plt.subplot(2, 2, 2)
		ax = sns.lineplot(
			x="epoch",
			y="accepted_ratio",
			hue="alpha",
			style="relation",
			data=data_dict[relation_name + "_vl"]
		)
		ax.set_title("Validation - accepted_ratio - " + relation_name)
		plt.subplot(2, 2, 4)
		ax = sns.lineplot(
			x="epoch",
			y="accuracy",
			hue="alpha",
			style="relation",
			data=data_dict[relation_name + "_vl"]
		)
		ax.set_title("Validation - accuracy - " + relation_name)

		plt.savefig(os.path.join(plot_path, name) + "_" + relation_name + ".png")
		plt.close()


def plot_relation_accuracy_data_foreach(cfg, data, path, name):
	plot_path = os.path.join(path, cfg.name, "relation_accuracy_foreach")
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	for i, relation in enumerate(["add", "sub", "mul"]):

		relation_data = data[data["relation"] == relation]

		sns.set_style("whitegrid")
		plt.figure(figsize=(20, 10))

		for j, entity in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):
			entity_data = relation_data[relation_data["entity"] == entity]

			plt.subplot(2, 10, j+1)
			ax = sns.lineplot(x="epoch", y="accepted_ratio", hue="alpha", style="relation", data=entity_data, legend=False)
			ax.set(xlabel=None, ylabel=None)
			ax.set_title(str(entity) + " - AR - " + relation)
			plt.subplot(2, 10, j+11, sharex=ax, sharey=ax)
			ax = sns.lineplot(x="epoch", y="accuracy", hue="alpha", style="relation", data=entity_data, legend=False)
			ax.set_title(str(entity) + " - ACC - " + relation)
			ax.set(xlabel=None, ylabel=None)

		plt.savefig(os.path.join(plot_path, name) + "_" + relation + ".png")
		plt.close()
