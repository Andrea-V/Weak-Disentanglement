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

	tr_data_ae  = tr_data[tr_data["type"].isin(["recon", "prior", "gaussian", "categorical", "relational"])]
	tr_data_gan = tr_data[tr_data["type"].isin(["adv_gen", "adv_disc"])]
	tr_data_scores = tr_data[tr_data["type"].isin(["gen_score", "prior_score", "inter_score"])]#, "grad_penalty"])]

	vl_data_ae  = vl_data[vl_data["type"].isin(["recon", "prior", "gaussian", "categorical", "relational"])]
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

				if cfg.current_dataset == "Shapes3D":
					x = x.view((-1, 3, 64, 64)).float().to(cfg.device)
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
			elif cfg.current_dataset == "Shapes3D":
				x_i = x_i.view((n_samples * 64, 64, 3))
				c_i = c_i.view((64, 64, 3))
			
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

	plt.imshow(x)

	plt.savefig(os.path.join(plot_path, name + "samples") + ".png")
	plt.close()
	
	plt.figure(figsize=(model.prior.n_classes, 2.5))

	plt.imshow(x_centroids)

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
				elif cfg.current_dataset == "Shapes3d":
					x = x.view((-1, 3, 64, 64)).float().to(cfg.device)
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
			"accepted_ratio_more_hue": None,
			"accuracy_more_hue": None,
			"accepted_ratio_less_hue": None,
			"accuracy_less_hue": None,
			"accepted_ratio_bigger": None,
			"accuracy_bigger": None,
			"accepted_ratio_smaller": None,
			"accuracy_smaller": None,
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
			y_pred_more_hue, y_pred_less_hue, y_pred_bigger, y_pred_smaller, y_pred_shape = [], [], [], [], []
			y_true_more_hue, y_true_less_hue, y_true_bigger, y_true_smaller, y_true_shape = [], [], [], [], []

			for i, batch in enumerate(dataset_iter):
				batch = batch.to(cfg.device)

				z_pred = model.prior(batch)

				z_pred_more_hue = z_pred[0]
				z_pred_less_hue = z_pred[1]
				z_pred_bigger = z_pred[2]
				z_pred_smaller = z_pred[3]
				z_pred_shape = z_pred[4]


				y_pred_more_hue.append(model.prior.classify(z_pred_more_hue, threshold=alpha).squeeze())
				y_pred_less_hue.append(model.prior.classify(z_pred_less_hue, threshold=alpha).squeeze())
				y_pred_bigger.append(model.prior.classify(z_pred_bigger, threshold=alpha).squeeze())
				y_pred_smaller.append(model.prior.classify(z_pred_smaller, threshold=alpha).squeeze())
				y_pred_shape.append(model.prior.classify(z_pred_shape, threshold=alpha).squeeze())

				y_true_more_hue.append(batch.more_hue_target_label.squeeze())
				y_true_less_hue.append(batch.less_hue_target_label.squeeze())
				y_true_bigger.append(batch.bigger_target_label.squeeze())
				y_true_smaller.append(batch.smaller_target_label.squeeze())
				y_true_shape.append(batch.shape_target_label.squeeze())

			y_true_more_hue = torch.cat(y_true_more_hue, dim=0)
			y_true_less_hue = torch.cat(y_true_less_hue, dim=0)
			y_true_bigger = torch.cat(y_true_bigger, dim=0)
			y_true_smaller = torch.cat(y_true_smaller, dim=0)
			y_true_shape = torch.cat(y_true_shape, dim=0)

			y_pred_more_hue = torch.cat(y_pred_more_hue, dim=0)
			y_pred_less_hue = torch.cat(y_pred_less_hue, dim=0)
			y_pred_bigger = torch.cat(y_pred_bigger, dim=0)
			y_pred_smaller = torch.cat(y_pred_smaller, dim=0)
			y_pred_shape = torch.cat(y_pred_shape, dim=0)

			accepted_y_pred_more_hue = y_pred_more_hue[y_pred_more_hue != -1]
			accepted_y_pred_less_hue = y_pred_less_hue[y_pred_less_hue != -1]
			accepted_y_pred_bigger = y_pred_bigger[y_pred_bigger != -1]
			accepted_y_pred_smaller = y_pred_smaller[y_pred_smaller != -1]
			accepted_y_pred_shape = y_pred_shape[y_pred_shape != -1]

			accepted_y_true_more_hue = y_true_more_hue[y_pred_more_hue != -1]
			accepted_y_true_less_hue = y_true_less_hue[y_pred_less_hue != -1]
			accepted_y_true_bigger = y_true_bigger[y_pred_bigger != -1]
			accepted_y_true_smaller = y_true_smaller[y_pred_smaller != -1]
			accepted_y_true_shape = y_true_shape[y_pred_shape != -1]

			accepted_ratio_more_hue = float(accepted_y_pred_more_hue.shape[0]) / float(y_pred_more_hue.shape[0])
			accepted_ratio_less_hue = float(accepted_y_pred_less_hue.shape[0]) / float(y_pred_less_hue.shape[0])
			accepted_ratio_bigger = float(accepted_y_pred_bigger.shape[0]) / float(y_pred_bigger.shape[0])
			accepted_ratio_smaller = float(accepted_y_pred_smaller.shape[0]) / float(y_pred_smaller.shape[0])
			accepted_ratio_shape = float(accepted_y_pred_shape.shape[0]) / float(y_pred_shape.shape[0])

			accepted_ratio_tot = (accepted_ratio_more_hue + accepted_ratio_less_hue + accepted_ratio_bigger + accepted_ratio_smaller + accepted_ratio_shape) / 5.

			try:
				accuracy_more_hue = float((accepted_y_pred_more_hue == accepted_y_true_more_hue).sum()) / float(accepted_y_true_more_hue.shape[0])
			except ZeroDivisionError:
				accuracy_more_hue = 0.0

			try:
				accuracy_less_hue = float((accepted_y_pred_less_hue == accepted_y_true_less_hue).sum()) / float(accepted_y_true_less_hue.shape[0])
			except ZeroDivisionError:
				accuracy_less_hue = 0.0

			try:
				accuracy_bigger = float((accepted_y_pred_bigger == accepted_y_true_bigger).sum()) / float(accepted_y_true_bigger.shape[0])
			except ZeroDivisionError:
				accuracy_bigger = 0.0

			try:
				accuracy_smaller = float((accepted_y_pred_smaller == accepted_y_true_smaller).sum()) / float(accepted_y_true_smaller.shape[0])
			except ZeroDivisionError:
				accuracy_smaller = 0.0

			try:
				accuracy_shape = float((accepted_y_pred_shape == accepted_y_true_shape).sum()) / float(accepted_y_true_shape.shape[0])
			except ZeroDivisionError:
				accuracy_shape = 0.0


			accuracy_tot = (accuracy_more_hue + accuracy_less_hue + accuracy_bigger + accuracy_smaller + accuracy_shape) / 5.

			results[alpha]["accepted_ratio_more_hue"] = accepted_ratio_more_hue
			results[alpha]["accuracy_more_hue"] = accuracy_more_hue
			results[alpha]["accepted_ratio_less_hue"] = accepted_ratio_less_hue
			results[alpha]["accuracy_less_hue"] = accuracy_less_hue
			results[alpha]["accepted_ratio_bigger"] = accepted_ratio_bigger
			results[alpha]["accuracy_bigger"] = accuracy_bigger
			results[alpha]["accepted_ratio_smaller"] = accepted_ratio_smaller
			results[alpha]["accuracy_smaller"] = accuracy_smaller
			results[alpha]["accepted_ratio_shape"] = accepted_ratio_shape
			results[alpha]["accuracy_shape"] = accuracy_shape
			results[alpha]["accepted_ratio_tot"] = accepted_ratio_tot
			results[alpha]["accuracy_tot"] = accuracy_tot

	return results

# entities_ids: set of fixed more_hue-term entitities to compute the accuracy
# other entities_ids: other entities to sample for less_hue term
def compute_relation_accuracy_foreach_entity(cfg, model, fixed_entities_ids, other_entities_ids):
	results = {}
	for entity in fixed_entities_ids:
		results[entity] = {}

		for alpha in cfg.eval.alpha_thresholds:
			results[entity][alpha] = {
				"accepted_ratio_more_hue": None,
				"accuracy_more_hue": None,
				"accepted_ratio_less_hue": None,
				"accuracy_less_hue": None,
				"accepted_ratio_bigger": None,
				"accuracy_bigger": None,
				"accepted_ratio_smaller": None,
				"accuracy_smaller": None,
				"accepted_ratio_shape": None,
				"accuracy_shape": None,
				"accepted_ratio_tot": None,
				"accuracy_tot": None
			}

		# replicate iterable for each alpha level
		dataset = (
			mod_dset.new_batch_of_dsprites_relations(cfg, 1024, model.prior, other_entities_ids, fix_more_hue=entity)
			for _ in range(10)
		)

		dataset_iters = tee(dataset, len(cfg.eval.alpha_thresholds))

		with torch.no_grad():
			for dataset_iter, alpha in zip(dataset_iters, cfg.eval.alpha_thresholds):
				y_pred_more_hue, y_pred_less_hue, y_pred_bigger, y_pred_smaller, y_pred_shape = [], [], [], [], []
				y_true_more_hue, y_true_less_hue, y_true_bigger, y_true_smaller, y_true_shape = [], [], [], [], []

				for i, batch in enumerate(dataset_iter):
					batch = batch.to(cfg.device)

					z_pred = model.prior(batch)

					z_pred_more_hue = z_pred[0]
					z_pred_less_hue = z_pred[1]
					z_pred_bigger = z_pred[2]
					z_pred_smaller = z_pred[3]
					z_pred_shape = z_pred[4]

					y_pred_more_hue.append(model.prior.classify(z_pred_more_hue, threshold=alpha))
					y_pred_less_hue.append(model.prior.classify(z_pred_less_hue, threshold=alpha))
					y_pred_bigger.append(model.prior.classify(z_pred_bigger, threshold=alpha))
					y_pred_smaller.append(model.prior.classify(z_pred_smaller, threshold=alpha))
					y_pred_shape.append(model.prior.classify(z_pred_shape, threshold=alpha))

					y_true_more_hue.append(batch.more_hue_target_label.squeeze())
					y_true_less_hue.append(batch.less_hue_target_label.squeeze())
					y_true_bigger.append(batch.bigger_target_label.squeeze())
					y_true_smaller.append(batch.smaller_target_label.squeeze())
					y_true_shape.append(batch.shape_target_label.squeeze())

				y_true_more_hue = torch.cat(y_true_more_hue, dim=0)
				y_true_less_hue = torch.cat(y_true_less_hue, dim=0)
				y_true_bigger = torch.cat(y_true_bigger, dim=0)
				y_true_smaller = torch.cat(y_true_smaller, dim=0)
				y_true_shape = torch.cat(y_true_shape, dim=0)

				y_pred_more_hue = torch.cat(y_pred_more_hue, dim=0)
				y_pred_less_hue = torch.cat(y_pred_less_hue, dim=0)
				y_pred_bigger = torch.cat(y_pred_bigger, dim=0)
				y_pred_smaller = torch.cat(y_pred_smaller, dim=0)
				y_pred_shape = torch.cat(y_pred_shape, dim=0)

				accepted_y_pred_more_hue = y_pred_more_hue[y_pred_more_hue != -1]
				accepted_y_pred_less_hue = y_pred_less_hue[y_pred_less_hue != -1]
				accepted_y_pred_bigger = y_pred_bigger[y_pred_bigger != -1]
				accepted_y_pred_smaller = y_pred_smaller[y_pred_smaller != -1]
				accepted_y_pred_shape = y_pred_shape[y_pred_shape != -1]

				accepted_y_true_more_hue = y_true_more_hue[y_pred_more_hue != -1]
				accepted_y_true_less_hue = y_true_less_hue[y_pred_less_hue != -1]
				accepted_y_true_bigger = y_true_bigger[y_pred_bigger != -1]
				accepted_y_true_smaller = y_true_smaller[y_pred_smaller != -1]
				accepted_y_true_shape = y_true_shape[y_pred_shape != -1]

				accepted_ratio_more_hue = float(accepted_y_pred_more_hue.shape[0]) / float(y_pred_more_hue.shape[0])
				accepted_ratio_less_hue = float(accepted_y_pred_less_hue.shape[0]) / float(y_pred_less_hue.shape[0])
				accepted_ratio_bigger = float(accepted_y_pred_bigger.shape[0]) / float(y_pred_bigger.shape[0])
				accepted_ratio_smaller = float(accepted_y_pred_smaller.shape[0]) / float(y_pred_smaller.shape[0])
				accepted_ratio_shape = float(accepted_y_pred_shape.shape[0]) / float(y_pred_shape.shape[0])

				accepted_ratio_tot = (accepted_ratio_more_hue + accepted_ratio_less_hue + accepted_ratio_bigger + accepted_ratio_smaller + accepted_ratio_shape) / 5.

				try:
					accuracy_more_hue = float((accepted_y_pred_more_hue == accepted_y_true_more_hue).sum()) / float(accepted_y_true_more_hue.shape[0])
				except ZeroDivisionError:
					accuracy_more_hue = 0.0

				try:
					accuracy_less_hue = float((accepted_y_pred_less_hue == accepted_y_true_less_hue).sum()) / float(accepted_y_true_less_hue.shape[0])
				except ZeroDivisionError:
					accuracy_less_hue = 0.0

				try:
					accuracy_bigger = float((accepted_y_pred_bigger == accepted_y_true_bigger).sum()) / float(accepted_y_true_bigger.shape[0])
				except ZeroDivisionError:
					accuracy_bigger = 0.0

				try:
					accuracy_smaller = float((accepted_y_pred_smaller == accepted_y_true_smaller).sum()) / float(accepted_y_true_smaller.shape[0])
				except ZeroDivisionError:
					accuracy_smaller = 0.0

				try:
					accuracy_shape = float((accepted_y_pred_shape == accepted_y_true_shape).sum()) / float(accepted_y_true_shape.shape[0])
				except ZeroDivisionError:
					accuracy_shape = 0.0

				accuracy_tot = (accuracy_more_hue + accuracy_less_hue + accuracy_bigger + accuracy_smaller + accuracy_shape) / 5.

				results[entity][alpha]["accepted_ratio_more_hue"] = accepted_ratio_more_hue
				results[entity][alpha]["accuracy_more_hue"] = accuracy_more_hue
				results[entity][alpha]["accepted_ratio_less_hue"] = accepted_ratio_less_hue
				results[entity][alpha]["accuracy_less_hue"] = accuracy_less_hue
				results[entity][alpha]["accepted_ratio_bigger"] = accepted_ratio_bigger
				results[entity][alpha]["accuracy_bigger"] = accuracy_bigger
				results[entity][alpha]["accepted_ratio_smaller"] = accepted_ratio_smaller
				results[entity][alpha]["accuracy_smaller"] = accuracy_smaller
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
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_more_hue"],
				"accuracy": accuracy[alpha_level]["accuracy_more_hue"],
				"relation": "more_hue",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_less_hue"],
				"accuracy": accuracy[alpha_level]["accuracy_less_hue"],
				"relation": "less_hue",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_bigger"],
				"accuracy": accuracy[alpha_level]["accuracy_bigger"],
				"relation": "bigger",
				"alpha": alpha_level,
				"dataset": dataset
			},
			ignore_index=True
		)
		df_rel_accuracy = df_rel_accuracy.append(
			{
				"fold": fold,
				"epoch": epoch,
				"accepted_ratio": accuracy[alpha_level]["accepted_ratio_smaller"],
				"accuracy": accuracy[alpha_level]["accuracy_smaller"],
				"relation": "smaller",
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
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_more_hue"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_more_hue"],
					"relation": "more_hue",
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
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_less_hue"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_less_hue"],
					"relation": "less_hue",
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
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_bigger"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_bigger"],
					"relation": "bigger",
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
					"accepted_ratio": accuracy[entity][alpha_level]["accepted_ratio_smaller"],
					"accuracy": accuracy[entity][alpha_level]["accuracy_smaller"],
					"relation": "smaller",
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

	tr_more_hue_data = tr_data[tr_data["relation"] == "more_hue"]
	tr_less_hue_data = tr_data[tr_data["relation"] == "less_hue"]
	tr_bigger_data = tr_data[tr_data["relation"] == "bigger"]
	tr_smaller_data = tr_data[tr_data["relation"] == "smaller"]
	tr_shape_data = tr_data[tr_data["relation"] == "shape"]
	tr_tot_data = tr_data[tr_data["relation"] == "tot"]

	vl_more_hue_data = vl_data[vl_data["relation"] == "more_hue"]
	vl_less_hue_data = vl_data[vl_data["relation"] == "less_hue"]
	vl_bigger_data = vl_data[vl_data["relation"] == "bigger"]
	vl_smaller_data = vl_data[vl_data["relation"] == "smaller"]
	vl_shape_data = vl_data[vl_data["relation"] == "shape"]
	vl_tot_data = vl_data[vl_data["relation"] == "tot"]

	data_dict = {
		"more_hue_tr": tr_more_hue_data,
		"less_hue_tr": tr_less_hue_data,
		"bigger_tr": tr_bigger_data,
		"smaller_tr": tr_smaller_data,
		"shape_tr": tr_shape_data,
		"more_hue_vl": vl_more_hue_data,
		"less_hue_vl": vl_less_hue_data,
		"bigger_vl": vl_bigger_data,
		"smaller_vl": vl_smaller_data,
		"shape_vl": vl_shape_data,
		"tot_tr": tr_tot_data,
		"tot_vl": vl_tot_data
	}

	for relation_name in ["more_hue", "less_hue", "bigger", "smaller", "shape", "tot"]:
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
		
