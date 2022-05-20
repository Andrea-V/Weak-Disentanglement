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
			for i, (x, y_1hot) in enumerate(dataset[ds]):

				if i > cfg.log.max_batch_to_plot:
					break

				if cfg.current_dataset == "dSprites":
					x = x.view((-1, 1, 64, 64)).float().to(cfg.device)
				else:
					raise ValueError("Unknown dataset.")

				#y_1hot = F.one_hot(y, num_classes=cfg.data.n_classes).to(cfg.device)
				label = np.array(cfg.data.class_names)[np.argmax(y_1hot, axis=1)]

				#y_prior = model.prior.categorical.sample((cfg.data.batch_size,)).to(cfg.device)

				z_prior = model.prior.sample(y_1hot)
				z_post = model.encoder(x) #, y_1hot)

				#y_prior = y_prior.argmax(dim=-1).cpu().numpy()
				#y_post = y_post.argmax(dim=-1).cpu().numpy()
				
				z_prior = z_prior.cpu().numpy() 
				z_post = z_post.cpu().numpy()
				#print(y.shape)
				y = np.argmax(y_1hot.cpu().numpy(), axis=1)
				#print(y.shape)
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
				posterior_z[ds]["label"].extend(list(label))
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

	sns.set_style("whitegrid")
	plt.figure(figsize=(20, 10))
	try:
		ax1 = plt.subplot(2, 2, 1)
		sns.kdeplot(x=prior_z_tr["z1"], y=prior_z_tr["z2"], ax=ax1, shade=False)
		ax1.set_title("TR: Latent space - prior")
	except:
		pass

	ax2 = plt.subplot(2, 2, 2, sharex=ax1, sharey=ax1)
	sns.scatterplot(x="z1", y="z2", hue="label", data=prior_z_tr, ax=ax2, legend=True)#, palette=palette_prior)
	ax2.set_title("TR: Latent space - prior")

	try:
		ax3 = plt.subplot(2, 2, 3)
		sns.kdeplot(x=posterior_z_tr["z1"], y=posterior_z_tr["z2"], ax=ax3, shade=False)
		ax3.set_title("TR: Latent space - posterior")
	except:
		pass

	ax4 = plt.subplot(2, 2, 4, sharex=ax3, sharey=ax3)
	sns.scatterplot(x="z1", y="z2", hue="label", data=posterior_z_tr, ax=ax4, legend=True)#, palette=palette_posterior)
	ax4.set_title("TR: Latent space - posterior")

	plt.savefig(os.path.join(plot_path, name) + ".png")
	plt.close()

def plot_adaptive_prior(cfg, model, n_samples, path, name):
	plot_path = os.path.join(path, cfg.name, "adaptive_prior")
	if not os.path.exists(plot_path):
		os.makedirs(plot_path)

	model_prior = model.prior

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
			elif cfg.current_dataset == "dSprites":
				x_i = x_i.view((n_samples * 64, 64))
				c_i = c_i.view((64, 64))
			else:
				raise ValueError("Unknown dataset")
			
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

	with torch.no_grad():
		for dataset_iter, alpha in zip(dataset_iters, cfg.eval.alpha_thresholds):
			#print("- alpha level:", alpha, "log:", np.log(alpha))
			y_preds = []
			ys = []

			for i, (x, y) in enumerate(dataset_iter):

				if cfg.current_dataset == "dSprites":
					x = x.view((-1, 1, 64, 64)).float().to(cfg.device)
				elif cfg.current_dataset == "CIFAR10":
					x = x.view((-1, 3, 32, 32)).float().to(cfg.device)
				elif cfg.current_dataset == "HWF":
					x = x.view((-1, 1, 45, 45)).float().to(cfg.device)
				else: #flatten input
					x = x.view((-1, cfg.model.x_dim)).float().to(cfg.device)

				y = torch.argmax(y, dim=1).to(cfg.device)

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

def compute_relation_accuracy(cfg, model, entities_ids):
	results = {}
	for alpha in cfg.eval.alpha_thresholds:
		results[alpha] = {
			"accepted_ratio_left": None,
			"accuracy_left": None,
			"accepted_ratio_right": None,
			"accuracy_right": None,
			"accepted_ratio_up": None,
			"accuracy_up": None,
			"accepted_ratio_down": None,
			"accuracy_down": None,
			"accepted_ratio_shape": None,
			"accuracy_shape": None,
			"accepted_ratio_tot": None,
			"accuracy_tot": None
		}

	# replicate iterable for each alpha level
	dataset = (
		mod_dset.new_batch_of_dsprites_relations(cfg, cfg.data.prior_batch_size, model.prior, entities_ids)
		for _ in range(10)
	)

	dataset_iters = tee(dataset, len(cfg.eval.alpha_thresholds))

	with torch.no_grad():
		for dataset_iter, alpha in zip(dataset_iters, cfg.eval.alpha_thresholds):
			y_pred_left, y_pred_right, y_pred_up, y_pred_down, y_pred_shape = [], [], [], [], []
			y_true_left, y_true_right, y_true_up, y_true_down, y_true_shape = [], [], [], [], []

			for i, batch in enumerate(dataset_iter):
				batch = batch.to(cfg.device)

				z_pred = model.prior(batch)

				z_pred_left = z_pred[0]
				z_pred_right = z_pred[1]
				z_pred_up = z_pred[2]
				z_pred_down = z_pred[3]
				z_pred_shape = z_pred[4]


				y_pred_left.append(model.prior.classify(z_pred_left, threshold=alpha).squeeze())
				y_pred_right.append(model.prior.classify(z_pred_right, threshold=alpha).squeeze())
				y_pred_up.append(model.prior.classify(z_pred_up, threshold=alpha).squeeze())
				y_pred_down.append(model.prior.classify(z_pred_down, threshold=alpha).squeeze())
				y_pred_shape.append(model.prior.classify(z_pred_shape, threshold=alpha).squeeze())

				y_true_left.append(batch.left_target_label.squeeze())
				y_true_right.append(batch.right_target_label.squeeze())
				y_true_up.append(batch.up_target_label.squeeze())
				y_true_down.append(batch.down_target_label.squeeze())
				y_true_shape.append(batch.shape_target_label.squeeze())

			y_true_left = torch.cat(y_true_left, dim=0)
			y_true_right = torch.cat(y_true_right, dim=0)
			y_true_up = torch.cat(y_true_up, dim=0)
			y_true_down = torch.cat(y_true_down, dim=0)
			y_true_shape = torch.cat(y_true_shape, dim=0)

			y_pred_left = torch.cat(y_pred_left, dim=0)
			y_pred_right = torch.cat(y_pred_right, dim=0)
			y_pred_up = torch.cat(y_pred_up, dim=0)
			y_pred_down = torch.cat(y_pred_down, dim=0)
			y_pred_shape = torch.cat(y_pred_shape, dim=0)

			accepted_y_pred_left = y_pred_left[y_pred_left != -1]
			accepted_y_pred_right = y_pred_right[y_pred_right != -1]
			accepted_y_pred_up = y_pred_up[y_pred_up != -1]
			accepted_y_pred_down = y_pred_down[y_pred_down != -1]
			accepted_y_pred_shape = y_pred_shape[y_pred_shape != -1]

			accepted_y_true_left = y_true_left[y_pred_left != -1]
			accepted_y_true_right = y_true_right[y_pred_right != -1]
			accepted_y_true_up = y_true_up[y_pred_up != -1]
			accepted_y_true_down = y_true_down[y_pred_down != -1]
			accepted_y_true_shape = y_true_shape[y_pred_shape != -1]

			accepted_ratio_left = float(accepted_y_pred_left.shape[0]) / float(y_pred_left.shape[0])
			accepted_ratio_right = float(accepted_y_pred_right.shape[0]) / float(y_pred_right.shape[0])
			accepted_ratio_up = float(accepted_y_pred_up.shape[0]) / float(y_pred_up.shape[0])
			accepted_ratio_down = float(accepted_y_pred_down.shape[0]) / float(y_pred_down.shape[0])
			accepted_ratio_shape = float(accepted_y_pred_shape.shape[0]) / float(y_pred_shape.shape[0])

			accepted_ratio_tot = (accepted_ratio_left + accepted_ratio_right + accepted_ratio_up + accepted_ratio_down + accepted_ratio_shape) / 5.

			try:
				accuracy_left = float((accepted_y_pred_left == accepted_y_true_left).sum()) / float(accepted_y_true_left.shape[0])
			except ZeroDivisionError:
				accuracy_left = 0.0

			try:
				accuracy_right = float((accepted_y_pred_right == accepted_y_true_right).sum()) / float(accepted_y_true_right.shape[0])
			except ZeroDivisionError:
				accuracy_right = 0.0

			try:
				accuracy_up = float((accepted_y_pred_up == accepted_y_true_up).sum()) / float(accepted_y_true_up.shape[0])
			except ZeroDivisionError:
				accuracy_up = 0.0

			try:
				accuracy_down = float((accepted_y_pred_down == accepted_y_true_down).sum()) / float(accepted_y_true_down.shape[0])
			except ZeroDivisionError:
				accuracy_down = 0.0

			try:
				accuracy_shape = float((accepted_y_pred_shape == accepted_y_true_shape).sum()) / float(accepted_y_true_shape.shape[0])
			except ZeroDivisionError:
				accuracy_shape = 0.0


			accuracy_tot = (accuracy_left + accuracy_right + accuracy_up + accuracy_down + accuracy_shape) / 5.

			results[alpha]["accepted_ratio_left"] = accepted_ratio_left
			results[alpha]["accuracy_left"] = accuracy_left
			results[alpha]["accepted_ratio_right"] = accepted_ratio_right
			results[alpha]["accuracy_right"] = accuracy_right
			results[alpha]["accepted_ratio_up"] = accepted_ratio_up
			results[alpha]["accuracy_up"] = accuracy_up
			results[alpha]["accepted_ratio_down"] = accepted_ratio_down
			results[alpha]["accuracy_down"] = accuracy_down
			results[alpha]["accepted_ratio_shape"] = accepted_ratio_shape
			results[alpha]["accuracy_shape"] = accuracy_shape
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
				"accepted_ratio_left": None,
				"accuracy_left": None,
				"accepted_ratio_right": None,
				"accuracy_right": None,
				"accepted_ratio_up": None,
				"accuracy_up": None,
				"accepted_ratio_down": None,
				"accuracy_down": None,
				"accepted_ratio_shape": None,
				"accuracy_shape": None,
				"accepted_ratio_tot": None,
				"accuracy_tot": None
			}

		# replicate iterable for each alpha level
		dataset = (
			mod_dset.new_batch_of_dsprites_relations(cfg, 1024, model.prior, other_entities_ids, fix_left=entity)
			for _ in range(10)
		)

		dataset_iters = tee(dataset, len(cfg.eval.alpha_thresholds))

		with torch.no_grad():
			for dataset_iter, alpha in zip(dataset_iters, cfg.eval.alpha_thresholds):
				y_pred_left, y_pred_right, y_pred_up, y_pred_down, y_pred_shape = [], [], [], [], []
				y_true_left, y_true_right, y_true_up, y_true_down, y_true_shape = [], [], [], [], []

				for i, batch in enumerate(dataset_iter):
					batch = batch.to(cfg.device)

					z_pred = model.prior(batch)

					z_pred_left = z_pred[0]
					z_pred_right = z_pred[1]
					z_pred_up = z_pred[2]
					z_pred_down = z_pred[3]
					z_pred_shape = z_pred[4]

					y_pred_left.append(model.prior.classify(z_pred_left, threshold=alpha))
					y_pred_right.append(model.prior.classify(z_pred_right, threshold=alpha))
					y_pred_up.append(model.prior.classify(z_pred_up, threshold=alpha))
					y_pred_down.append(model.prior.classify(z_pred_down, threshold=alpha))
					y_pred_shape.append(model.prior.classify(z_pred_shape, threshold=alpha))

					y_true_left.append(batch.left_target_label.squeeze())
					y_true_right.append(batch.right_target_label.squeeze())
					y_true_up.append(batch.up_target_label.squeeze())
					y_true_down.append(batch.down_target_label.squeeze())
					y_true_shape.append(batch.shape_target_label.squeeze())

				y_true_left = torch.cat(y_true_left, dim=0)
				y_true_right = torch.cat(y_true_right, dim=0)
				y_true_up = torch.cat(y_true_up, dim=0)
				y_true_down = torch.cat(y_true_down, dim=0)
				y_true_shape = torch.cat(y_true_shape, dim=0)

				y_pred_left = torch.cat(y_pred_left, dim=0)
				y_pred_right = torch.cat(y_pred_right, dim=0)
				y_pred_up = torch.cat(y_pred_up, dim=0)
				y_pred_down = torch.cat(y_pred_down, dim=0)
				y_pred_shape = torch.cat(y_pred_shape, dim=0)

				accepted_y_pred_left = y_pred_left[y_pred_left != -1]
				accepted_y_pred_right = y_pred_right[y_pred_right != -1]
				accepted_y_pred_up = y_pred_up[y_pred_up != -1]
				accepted_y_pred_down = y_pred_down[y_pred_down != -1]
				accepted_y_pred_shape = y_pred_shape[y_pred_shape != -1]

				accepted_y_true_left = y_true_left[y_pred_left != -1]
				accepted_y_true_right = y_true_right[y_pred_right != -1]
				accepted_y_true_up = y_true_up[y_pred_up != -1]
				accepted_y_true_down = y_true_down[y_pred_down != -1]
				accepted_y_true_shape = y_true_shape[y_pred_shape != -1]

				accepted_ratio_left = float(accepted_y_pred_left.shape[0]) / float(y_pred_left.shape[0])
				accepted_ratio_right = float(accepted_y_pred_right.shape[0]) / float(y_pred_right.shape[0])
				accepted_ratio_up = float(accepted_y_pred_up.shape[0]) / float(y_pred_up.shape[0])
				accepted_ratio_down = float(accepted_y_pred_down.shape[0]) / float(y_pred_down.shape[0])
				accepted_ratio_shape = float(accepted_y_pred_shape.shape[0]) / float(y_pred_shape.shape[0])

				accepted_ratio_tot = (accepted_ratio_left + accepted_ratio_right + accepted_ratio_up + accepted_ratio_down + accepted_ratio_shape) / 5.

				try:
					accuracy_left = float((accepted_y_pred_left == accepted_y_true_left).sum()) / float(accepted_y_true_left.shape[0])
				except ZeroDivisionError:
					accuracy_left = 0.0

				try:
					accuracy_right = float((accepted_y_pred_right == accepted_y_true_right).sum()) / float(accepted_y_true_right.shape[0])
				except ZeroDivisionError:
					accuracy_right = 0.0

				try:
					accuracy_up = float((accepted_y_pred_up == accepted_y_true_up).sum()) / float(accepted_y_true_up.shape[0])
				except ZeroDivisionError:
					accuracy_up = 0.0

				try:
					accuracy_down = float((accepted_y_pred_down == accepted_y_true_down).sum()) / float(accepted_y_true_down.shape[0])
				except ZeroDivisionError:
					accuracy_down = 0.0

				try:
					accuracy_shape = float((accepted_y_pred_shape == accepted_y_true_shape).sum()) / float(accepted_y_true_shape.shape[0])
				except ZeroDivisionError:
					accuracy_shape = 0.0

				accuracy_tot = (accuracy_left + accuracy_right + accuracy_up + accuracy_down + accuracy_shape) / 5.

				results[entity][alpha]["accepted_ratio_left"] = accepted_ratio_left
				results[entity][alpha]["accuracy_left"] = accuracy_left
				results[entity][alpha]["accepted_ratio_right"] = accepted_ratio_right
				results[entity][alpha]["accuracy_right"] = accuracy_right
				results[entity][alpha]["accepted_ratio_up"] = accepted_ratio_up
				results[entity][alpha]["accuracy_up"] = accuracy_up
				results[entity][alpha]["accepted_ratio_down"] = accepted_ratio_down
				results[entity][alpha]["accuracy_down"] = accuracy_down
				results[entity][alpha]["accepted_ratio_shape"] = accepted_ratio_shape
				results[entity][alpha]["accuracy_shape"] = accuracy_shape
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
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_left"],
				"accuracy": accuracy[alpha_level]["accuracy_left"],
				"relation": "left",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_right"],
				"accuracy": accuracy[alpha_level]["accuracy_right"],
				"relation": "right",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_up"],
				"accuracy": accuracy[alpha_level]["accuracy_up"],
				"relation": "up",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_down"],
				"accuracy": accuracy[alpha_level]["accuracy_down"],
				"relation": "down",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_shape"],
				"accuracy": accuracy[alpha_level]["accuracy_shape"],
				"relation": "shape",
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
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_left"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_left"],
					"relation": "left",
					"alpha": alpha_level,
					"dataset": dataset,
					"entity": entity
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_right"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_right"],
					"relation": "right",
					"alpha": alpha_level,
					"dataset": dataset,
					"entity": entity
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_up"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_up"],
					"relation": "up",
					"alpha": alpha_level,
					"dataset": dataset,
					"entity": entity
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_down"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_down"],
					"relation": "down",
					"alpha": alpha_level,
					"dataset": dataset,
					"entity": entity
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_shape"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_shape"],
					"relation": "shape",
					"alpha": alpha_level,
					"dataset": dataset,
					"entity": entity
				},
				ignore_index=True
			)
			df_rel_accuracy_foreach = df_rel_accuracy_foreach.append(
				{
					"fold": fold,
					"epoch": epoch,
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_tot"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_tot"],
					"relation": "tot",
					"alpha": alpha_level,
					"dataset": dataset,
					"entity": entity
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

	tr_left_data = tr_data[tr_data["relation"] == "left"]
	tr_right_data = tr_data[tr_data["relation"] == "right"]
	tr_up_data = tr_data[tr_data["relation"] == "up"]
	tr_down_data = tr_data[tr_data["relation"] == "down"]
	tr_shape_data = tr_data[tr_data["relation"] == "shape"]
	tr_tot_data = tr_data[tr_data["relation"] == "tot"]

	vl_left_data = vl_data[vl_data["relation"] == "left"]
	vl_right_data = vl_data[vl_data["relation"] == "right"]
	vl_up_data = vl_data[vl_data["relation"] == "up"]
	vl_down_data = vl_data[vl_data["relation"] == "down"]
	vl_shape_data = vl_data[vl_data["relation"] == "shape"]
	vl_tot_data = vl_data[vl_data["relation"] == "tot"]

	data_dict = {
		"left_tr": tr_left_data,
		"right_tr": tr_right_data,
		"up_tr": tr_up_data,
		"down_tr": tr_down_data,
		"shape_tr": tr_shape_data,
		"left_vl": vl_left_data,
		"right_vl": vl_right_data,
		"up_vl": vl_up_data,
		"down_vl": vl_down_data,
		"shape_vl": vl_shape_data,
		"tot_tr": tr_tot_data,
		"tot_vl": vl_tot_data
	}

	for relation_name in ["left", "right", "up", "down", "shape", "tot"]:
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
		
