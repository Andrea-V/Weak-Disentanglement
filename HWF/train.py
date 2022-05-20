import config
import torch
from torch import nn
from model import *
import loss
import dataset
import log
import pandas as pd
from sklearn.model_selection import RepeatedKFold
import pprint as pp
import prior
from time import sleep
import sys
from torch_geometric.data import Data

#torch.set_printoptions(threshold=5000)

def odd_even(y): 
	y = (y % 2 == 1).int()
	return y

def identity(a):
	return a

def set_trainable(trainables):
	for model, trainable in trainables.items():
		if trainable:
			for p in model.parameters():
				p.requires_grad = True
		else:
			for p in model.parameters():
				p.requires_grad = False

def run_epoch(cfg, data_iter, n_batches, model, losses):
	total_loss = {
		"recon": 0,
		"adv_disc": 0,
		"adv_gen": 0,
		"prior": 0,
		"gen_score": 0,
		"prior_score": 0,
		"grad_penalty": 0,
		"gaussian": 0,
		"categorical": 0

	}

	for it, (x, y, label) in enumerate(data_iter):

		#if it > 1:
		#	break

		#print("-- Iteration:", it+1, "/", n_batches)
		x = x.to(cfg.device)
		y = y.to(cfg.device)

		# drop labels
		y = torch.zeros((cfg.data.batch_size, cfg.data.n_classes,), dtype=torch.long).to(cfg.device)

		# reconstruction phase
		#print("--- AE phase")
		set_trainable({
			model.prior: False,
			model.encoder: True,
			model.decoder: True,
			model.discriminator: False
		})
		#with torch.autograd.set_detect_anomaly(True):
		x_rec, _ = model(x=x, phase="ae")
		recon_loss = losses["reconstruction"](x_rec, x)

		if cfg.train.phase != "warmup":
			# prior adaptation phase
			#print("--- PRIOR phase")
			set_trainable({
				model.prior: True,
				model.encoder: False,
				model.decoder: False,
				model.discriminator: False
			})
			if model.training: # training batch
				batch = dataset.new_batch_of_hwf_relations(
					cfg, cfg.data.prior_batch_size, model.prior,
					[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
				).to(cfg.device)
			else: # validation batch
				batch = dataset.new_batch_of_hwf_relations(
					cfg, cfg.data.prior_batch_size, model.prior,
					[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
				).to(cfg.device)


			z_true_add = batch.add_target_sample
			z_true_sub = batch.sub_target_sample
			z_true_mul = batch.mul_target_sample
			z_pred_add, z_pred_sub, z_pred_mul = model(relational_batch=batch, phase="prior")
			prior_loss = losses["prior"](z_true_add, z_pred_add, z_true_sub, z_pred_sub, z_true_mul, z_pred_mul)

		else:
			prior_loss = torch.Tensor([0.])

		# adversarial phase - gen
		#print("--- GAN_GEN phase")
		set_trainable({
			model.prior: False,
			model.encoder: True,
			model.decoder: False,
			model.discriminator: False
		})
		z_gen_score_g = model(x=x, y=y, phase="gan_gen")
		adv_loss_gen = losses["gan_gen"](z_gen_score_g)

		# adversarial phase - disc
		#print("--- GAN_DISC phase")
		set_trainable({
			model.prior: False,
			model.encoder: False,
			model.decoder: False,
			model.discriminator: True
		})
		z_gen_score, z_prior_score, z_gen, z_prior = model(x=x, y=y, phase="gan_disc")
		adv_loss_disc, grad_penalty = losses["gan_disc"](z_gen_score, z_prior_score, z_gen, z_prior)

		z_gen_score = torch.sigmoid(z_gen_score)
		z_prior_score = torch.sigmoid(z_prior_score)

		total_loss["recon"] += recon_loss.item()
		total_loss["adv_disc"] += adv_loss_disc.item()
		total_loss["adv_gen"] += adv_loss_gen.item()
		total_loss["prior"] += prior_loss.mean().item()
		total_loss["gen_score"] += z_gen_score.mean().item()
		total_loss["prior_score"] += z_prior_score.mean().item()
		total_loss["grad_penalty"] += 0 #grad_penalty.item()
		
	for key in total_loss:
		total_loss[key] /= n_batches

	return total_loss

# moved to the top level bc otherwise pickle won't work
def sigmoid(x):
	return torch.sigmoid(x)
def tanh(x):
	return torch.tanh(x)
def relu(x):
	return torch.relu(x)
def linear(x):
	return x


def train_model(cfg, fold, tr_set, vl_set):
	print("- Init model...")

	# tanh   = lambda x: torch.tanh(x)
	# relu   = lambda x: torch.relu(x)
	# linear = lambda x: x

	model = SemiSupervisedAbstractionAutoencoder(
		cfg = cfg,
		encoder=Conv45Encoder(nc=cfg.data.n_channels, nf=cfg.model.nf_dim, nz=cfg.model.z_dim),
		decoder=Conv45Decoder(nc=cfg.data.n_channels, nf=cfg.model.nf_dim, nz=cfg.model.z_dim),
		discriminator=MLP(
			cfg,
			3,
			(cfg.model.z_dim + cfg.data.n_classes, 1024, 1),
			activation=tanh, out_activation=linear
		),
		prior=prior.Uniform(cfg, z_dim=cfg.model.z_dim, low=-1., high=1.)
	)

	#for p in model.parameters():
	#	#if p.dim() > 1:
	#	torch.nn.init.normal_(p, mean=0.0, std=0.02)

	print("- Init loss and optimizers...")
	model.to(cfg.device)

	reconstruction_criterion = nn.BCEWithLogitsLoss().to(cfg.device)
	gan_gen_criterion = nn.BCEWithLogitsLoss().to(cfg.device)
	gan_disc_criterion = nn.BCEWithLogitsLoss().to(cfg.device)
	prior_criterion = loss.RelationalLoss(model.prior).to(cfg.device)

	dec_optim = torch.optim.Adam(model.decoder.parameters(), lr=cfg.train.lr.dec)
	enc_optim = torch.optim.Adam(model.encoder.parameters(), lr=cfg.train.lr.enc)
	disc_optim = torch.optim.Adam(model.discriminator.parameters(), lr=cfg.train.lr.disc)
	prior_optim = torch.optim.Adam(model.prior.parameters(), lr=cfg.train.lr.prior)

	#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=10, factor=0.5, verbose=True)

	# loss computes:
	tr_losses = {
		"reconstruction": loss.ReconstructionLossCompute(cfg, reconstruction_criterion, enc_optim, dec_optim, train=True),
		"gan_gen": loss.GeneratorLossCompute(cfg, gan_gen_criterion, enc_optim, train=cfg.loss.adversarial_train),
		"gan_disc": loss.DiscriminatorLossCompute(cfg, gan_disc_criterion, disc_optim, train=cfg.loss.adversarial_train),
		"prior": loss.RelationalLossCompute(cfg, prior_criterion, prior_optim, train=False)
	}
	vl_losses = {
		"reconstruction": loss.ReconstructionLossCompute(cfg, reconstruction_criterion, enc_optim, dec_optim, train=False),
		"gan_gen": loss.GeneratorLossCompute(cfg, gan_gen_criterion, enc_optim, train=False),
		"gan_disc": loss.DiscriminatorLossCompute(cfg, gan_disc_criterion, disc_optim, train=False),
		"prior": loss.RelationalLossCompute(cfg, prior_criterion, prior_optim, train=False)
	}

	print("- Init training loop...")
	df_loss = pd.DataFrame()
	df_accuracy = pd.DataFrame()
	df_rel_accuracy = pd.DataFrame()
	df_rel_accuracy_foreach = pd.DataFrame()

	for epoch in range(cfg.train.n_epochs):
		print("- epoch:", epoch + 1, "/", cfg.train.n_epochs)

		if epoch == cfg.train.warmup:
			print("-- Start prior training...")
			model.prior = prior.AdaptiveRelationalPrior(
				cfg,
				cfg.model.z_dim,
				cfg.data.n_classes
			).to(cfg.device)
			model.prior.init_adaptive_prior((batch for batch in tr_set), model)
			prior_optim = torch.optim.Adam(model.prior.parameters(), lr=cfg.train.lr.prior)
			cfg.train.phase = "relational"
			tr_losses["prior"] = loss.RelationalLossCompute(cfg, prior_criterion, prior_optim, train=True)
			vl_losses["prior"] = loss.RelationalLossCompute(cfg, prior_criterion, prior_optim, train=False)

			print("- New phase:", cfg.train.phase)

		model.train()
		tr_loss = run_epoch(cfg, (batch for batch in tr_set), len(tr_set), model, losses=tr_losses)

		if vl_set is not None:
			with torch.no_grad():
				model.eval()
				vl_loss = run_epoch(cfg, (batch for batch in vl_set), len(vl_set), model, losses=vl_losses)
		else:
			vl_loss = {
				"recon": 0,
				"adv_disc": 0,
				"adv_gen": 0,
				"prior": 0,
				"gen_score": 0,
				"prior_score": 0
			}
		#scheduler.step(ts_loss)
		
		#print("- TR loss:")
		#pp.pprint(tr_loss)
		#print("- VL loss:")
		#pp.pprint(vl_loss)

		for key in tr_loss:
			df_loss = df_loss.append({"fold": fold, "epoch": epoch, "loss": tr_loss[key], "type": key, "dataset": "training"}, ignore_index=True)	
			df_loss = df_loss.append({"fold": fold, "epoch": epoch, "loss": vl_loss[key], "type": key, "dataset": "validation"}, ignore_index=True)

		print("- Plot training data...")
		log.plot_training_data(cfg, df_loss, cfg.log.plots_path, "fold_" + str(fold) + "_training_curve")

		if epoch % cfg.train.store_interval == 0:
			if cfg.train.store_model:
				print("- Store model...")
				log.store_model(cfg, model, cfg.log.store_path, cfg.name + "_epoch_" + str(epoch))

		if epoch % cfg.train.log_interval == 0:
			print("- Store training data...")
			log.store_training_data(
				cfg,
				df_loss.to_dict(),
				cfg.log.plots_path,
				"fold_" + str(fold) + "_training_data"
			)
			print("- Plot latent space...")
			log.plot_latent_space_abs(
				cfg,
				model,
				(batch for batch in tr_set),
				(batch for batch in vl_set),
				cfg.log.plots_path,
				"fold_" + str(fold) + "_epoch_" + str(epoch) +"_latent_space"
			)
			if epoch > cfg.train.warmup:
				if cfg.log.adaptive_prior:
					print("- Store adaptive prior...")
					log.store_adaptive_prior(
						cfg, model,
						cfg.log.store_path,
						str(fold) + "_" + cfg.name + "_prior_" + str(epoch)
					)
					print("- Plot adaptive prior...")
					log.plot_adaptive_prior(
						cfg,
						model, 16,
						cfg.log.plots_path,
						"fold_" + str(fold) + "_epoch_" + str(epoch) +"_ap"
					)
			if epoch > cfg.train.warmup:
				if cfg.log.clustering_accuracy:
					print("- Compute clustering accuracy...")
					tr_accuracy = log.compute_clustering_accuracy(cfg, model, (batch for batch in tr_set))
					vl_accuracy = log.compute_clustering_accuracy(cfg, model, (batch for batch in vl_set))

					print("- Adapt clustering accuracy...")
					df_accuracy = log.adapt_accuracy_to_dataframe(fold, epoch, tr_accuracy, df_accuracy, "training")
					df_accuracy = log.adapt_accuracy_to_dataframe(fold, epoch, vl_accuracy, df_accuracy, "validation")

					print("- Store clustering accuracy...")
					log.store_accuracy_data(cfg, df_accuracy.to_dict(), cfg.log.plots_path,"fold_" + str(fold) + "_accuracy_data")

					print("- Plot clustering accuracy...")
					log.plot_accuracy_data(cfg, df_accuracy, cfg.log.plots_path, "fold_" + str(fold) + "_accuracy_curve")

				if cfg.log.relation_accuracy:
					print("- Compute relation accuracy (interpolation)")
					tr_relation_accuracy = log.compute_relation_accuracy(
						cfg,
						model,
						entities_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
					)
					vl_relation_accuracy = log.compute_relation_accuracy(
						cfg,
						model,
						entities_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
					)
					print("- Adapt relation accuracy (interpolation)")
					df_rel_accuracy = log.adapt_relation_accuracy_to_dataframe(
						fold,
						epoch,
						tr_relation_accuracy,
						df_rel_accuracy,
						"training"
					)
					df_rel_accuracy = log.adapt_relation_accuracy_to_dataframe(
						fold,
						epoch,
						vl_relation_accuracy,
						df_rel_accuracy,
						"validation"
					)
					print("- Store relation accuracy (interpolation)")
					log.store_relation_accuracy_data(
						cfg, df_rel_accuracy.to_dict(),
						cfg.log.plots_path,
						"fold_" + str(fold) + "_relation_accuracy_data"
					)

					print("- Plot relation accuracy (interpolation)")
					log.plot_relation_accuracy_data(
						cfg, df_rel_accuracy,
						cfg.log.plots_path,
						"fold_" + str(fold) + "_relation_accuracy_data"
					)

	return model, df_loss

def start_training_thread(cfg):

	print("New model training: ", cfg.name)

	print("***", cfg.name, "***")
	print("- Current dataset:", cfg.current_dataset)
	print("- Loading dataset...", cfg.data.path)
	
	if cfg.current_dataset == "HWF":
		tr_set, vl_set, ts_set = dataset.load_hwf(cfg, cfg.data.path, cfg.data.batch_size)
	else:
		raise ValueError("Unknown dataset.")

	print("* Data loaded:", cfg.current_dataset, "*")
	print("- Batch size:", cfg.data.batch_size)
	print("- Training set size (batches):", len(tr_set))
	print("- Validation set size (batches):", len(vl_set))
	print("- Test set size (batches):", len(ts_set))
	
	#with torch.autograd.set_detect_anomaly(True):
	model, df_loss = train_model(cfg, fold, tr_set, vl_set)
	
	df_loss_cv = pd.concat([df_loss, df_loss_cv], ignore_index=True)

	if cfg.train.store_model:
		log.store_model(cfg, model, cfg.log.store_path, cfg.name)
    
    print("ALL GOOD")

if __name__ == '__main__':
	start_training_thread(config.make_config(config.config))
