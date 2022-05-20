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
    "name": "dSprites,z=8,final",
    "current_dataset": "dSprites",
    "data": {
        "dSprites": {
            "prior_batch_size": 128,
            "batch_size": 1024,
            "path": "./data/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz",
            "n_workers_tr": 4,
            "n_workers_vl": 4,
            "n_workers_ts": 4,
            "n_classes": 27, # 3 x_position ^ 3 y_position ^ 3 shapes
            "n_entities": 27,
            "n_channels": 1,
            "class_names": [
                "<^S", "<xS", "<vS", "x^S", "xxS", "xvS", ">^S", ">xS", ">vS",
                "<^E", "<xE", "<vE", "x^E", "xxE", "xvE", ">^E", ">xE", ">vE",
                "<^H", "<xH", "<vH", "x^H", "xxH", "xvH", ">^H", ">xH", ">vH"
            ],
        }
    },
    "model": {
        "z_dim": 8,
        "nf_dim": 32,
        "n_relational_layers": 3
    },
    "prior": {
        "supervision_amount": 20,
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
        "gen_threshold": None,
        "warmup": 1000,
        "n_epochs": 6000,
        "cv_n_folds": 3,
        "cv_n_repeats": 3,
        "store_model": False,
        "log_interval": 500,
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
        "relation_accuracy_foreach": False,
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
