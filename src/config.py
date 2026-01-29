r_config = {
    "batch_size": 32,
    "num_workers": 4,
    "threshold": 0.5,
    "alpha": 0.5,

    "unfreeze_from": "layer3",
    
    "num_epochs": 40,

    "optimizer": {
        "lr": 1e-4,
        "weight_decay": 5e-4
    },

    "scheduler": {
        "factor": 0.1,
        "patience": 3
    },

    "early_stopping": {
        "patience": 7
    }
}


e_config = {
    "batch_size": 32,
    "num_workers": 4,
    "threshold": 0.5,
    "alpha": 0.5,

    "unfreeze_last_n_blocks": 3,
    
    "num_epochs": 40,

    "optimizer": {
        "lr": 3e-5,
        "weight_decay": 1e-4
    },

    "scheduler": {
        "factor": 0.1,
        "patience": 3
    },

    "early_stopping": {
        "patience": 8
    }
}


ensemble_config = {
    "alpha_candidates": [i / 10 for i in range(11)]
}
