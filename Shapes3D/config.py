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
    "name": "Shapes3d,z=8,test",
    "current_dataset": "Shapes3D",
    "data": {
        "Shapes3D": {
            "prior_batch_size": 128,
            "batch_size": 1024,
            "path": "./data/3d-shapes-master/3dshapes.h5",
            "n_workers_tr": 4,
            "n_workers_vl": 4,
            "n_workers_ts": 4,
            "n_entities": 120,
            "n_channels": 1,
            "class_names": [
                "HU0,SH0,SC0", "HU0,SH0,SC1", "HU0,SH0,SC2",
                "HU0,SH1,SC0", "HU0,SH1,SC1", "HU0,SH1,SC2",
                "HU0,SH2,SC0", "HU0,SH2,SC1", "HU0,SH2,SC2",
                "HU0,SH3,SC0", "HU0,SH3,SC1", "HU0,SH3,SC2",

                "HU1,SH0,SC0", "HU1,SH0,SC1", "HU1,SH0,SC2",
                "HU1,SH1,SC0", "HU1,SH1,SC1", "HU1,SH1,SC2",
                "HU1,SH2,SC0", "HU1,SH2,SC1", "HU1,SH2,SC2",
                "HU1,SH3,SC0", "HU1,SH3,SC1", "HU1,SH3,SC2",

                "HU2,SH0,SC0", "HU2,SH0,SC1", "HU2,SH0,SC2",
                "HU2,SH1,SC0", "HU2,SH1,SC1", "HU2,SH1,SC2",
                "HU2,SH2,SC0", "HU2,SH2,SC1", "HU2,SH2,SC2",
                "HU2,SH3,SC0", "HU2,SH3,SC1", "HU2,SH3,SC2",

                "HU3,SH0,SC0", "HU3,SH0,SC1", "HU3,SH0,SC2",
                "HU3,SH1,SC0", "HU3,SH1,SC1", "HU3,SH1,SC2",
                "HU3,SH2,SC0", "HU3,SH2,SC1", "HU3,SH2,SC2",
                "HU3,SH3,SC0", "HU3,SH3,SC1", "HU3,SH3,SC2",

                "HU4,SH0,SC0", "HU4,SH0,SC1", "HU4,SH0,SC2",
                "HU4,SH1,SC0", "HU4,SH1,SC1", "HU4,SH1,SC2",
                "HU4,SH2,SC0", "HU4,SH2,SC1", "HU4,SH2,SC2",
                "HU4,SH3,SC0", "HU4,SH3,SC1", "HU4,SH3,SC2",

                "HU5,SH0,SC0", "HU5,SH0,SC1", "HU5,SH0,SC2",
                "HU5,SH1,SC0", "HU5,SH1,SC1", "HU5,SH1,SC2",
                "HU5,SH2,SC0", "HU5,SH2,SC1", "HU5,SH2,SC2",
                "HU5,SH3,SC0", "HU5,SH3,SC1", "HU5,SH3,SC2",

                "HU6,SH0,SC0", "HU6,SH0,SC1", "HU6,SH0,SC2",
                "HU6,SH1,SC0", "HU6,SH1,SC1", "HU6,SH1,SC2",
                "HU6,SH2,SC0", "HU6,SH2,SC1", "HU6,SH2,SC2",
                "HU6,SH3,SC0", "HU6,SH3,SC1", "HU6,SH3,SC2",

                "HU7,SH0,SC0", "HU7,SH0,SC1", "HU7,SH0,SC2",
                "HU7,SH1,SC0", "HU7,SH1,SC1", "HU7,SH1,SC2",
                "HU7,SH2,SC0", "HU7,SH2,SC1", "HU7,SH2,SC2",
                "HU7,SH3,SC0", "HU7,SH3,SC1", "HU7,SH3,SC2",

                "HU8,SH0,SC0", "HU8,SH0,SC1", "HU8,SH0,SC2",
                "HU8,SH1,SC0", "HU8,SH1,SC1", "HU8,SH1,SC2",
                "HU8,SH2,SC0", "HU8,SH2,SC1", "HU8,SH2,SC2",
                "HU8,SH3,SC0", "HU8,SH3,SC1", "HU8,SH3,SC2",

                "HU9,SH0,SC0", "HU9,SH0,SC1", "HU9,SH0,SC2",
                "HU9,SH1,SC0", "HU9,SH1,SC1", "HU9,SH1,SC2",
                "HU9,SH2,SC0", "HU9,SH2,SC1", "HU9,SH2,SC2",
                "HU9,SH3,SC0", "HU9,SH3,SC1", "HU9,SH3,SC2",
            ],
        }
    },
    "model": {
        "z_dim": 16,
        "nf_dim": 64,
        "n_relations": 5,
        "n_relational_layers": 3
    },
    "prior": {
        "supervision_amount": 30,
        "gm_cov": 0.1
    },
    "loss": {
        "gp_weight": 10,
        "adversarial_weight": 0.2,
        "prior_weight": 1.,
        "adversarial_train": False
    },
    "train": {
        "phase": "warmup",
        "warmup": 5000,
        "n_epochs": 15001,
        "cv_n_folds": 3,
        "cv_n_repeats": 3,
        "store_model": False,
        "log_interval": 500,
        "store_interval": 1000,
        "lr": {
            "dec": 1e-5,
            "disc": 1e-5,
            "enc": 1e-5,
            "prior": 1e-5
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
