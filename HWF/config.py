from types import SimpleNamespace
import gpustat
import os
from sys import exit

def best_gpu():
    stats = gpustat.GPUStatCollection.new_query()
    ids = map(lambda gpu: int(gpu.entry['index']), stats)
    ratios = map(lambda gpu: float(gpu.entry['memory.used']) / float(gpu.entry['memory.total']), stats)
    bestGPU = min(zip(ids, ratios), key=lambda x: x[1])[0]
    return bestGPU


# selects dataset that is currently used
config = {
    "device": best_gpu(),
    "name": "HWF,z=2,figures",
    "current_dataset": "HWF",
    "data": {
        "HWF": {
            "prior_batch_size": 256,
            "batch_size": 1024,
            "path": "./data/HWF/",
            "n_workers_tr": 0,
            "n_workers_vl": 0,
            "n_workers_ts": 0,
            "n_classes": 13, # 13 bc we do not consider "div"
            "n_entities": 10,
            "n_channels": 1,
            "class_names": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "*"],
        }
    },
    "model": {
        "z_dim": 8,
        "nf_dim": 32,
        "n_relational_layers": 3
    },
    "prior": {
        "supervision_amount": 20,  # number of labelled samples to use for init adptive prior
        "gm_cov": 0.1
    },
    "loss": {
        "gp_weight": 10,
        "adversarial_weight": 0.2,
        "prior_weight": 1.,
        "adversarial_train": True
    },
    "train": {
        "phase": "warmup",
        "drop_label_amount": 0., # labels < of thresold are dropped
        "gen_threshold": None,  # training sample will be {0...gen_threh} + N2, validation {gen_threh....N_max} + N2
        "warmup": 1000,
        "n_epochs": 6000,
        "cv_n_folds": 3,
        "cv_n_repeats": 3,
        "store_model": False,
        "log_interval": 100,
        "store_interval": 1000,
        "lr": {
            "dec": 1e-4,
            "disc": 1e-4,
            "enc": 1e-4,
            "prior": 1e-4
        }
    },
    "log": {
        "adaptive_prior": True,
        "clustering_accuracy": True,
        "relation_accuracy": True,
        "relation_accuracy_foreach": True,
        "max_batch_to_plot": 5,
        "store_path": "./models",
        "plots_path": "./plots"
    }
}

def make_config(config):
    config["data"] = config["data"][config["current_dataset"]]
    config["train"]["lr"] = SimpleNamespace(**config["train"]["lr"])
    config["data"] = SimpleNamespace(**config["data"])
    config["model"] = SimpleNamespace(**config["model"])
    config["loss"] = SimpleNamespace(**config["loss"])
    config["train"] = SimpleNamespace(**config["train"])
    config["log"] = SimpleNamespace(**config["log"])
    config["prior"] = SimpleNamespace(**config["prior"])
    config = SimpleNamespace(**config)
    return config
