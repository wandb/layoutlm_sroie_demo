config = {
    "tokenizer_name": "microsoft/layoutlm-base-uncased",
    "model_name": "microsoft/layoutlm-base-uncased",
    "data_raw_path": "SROIE",
    "data_path": "data",
    "n_samples": None,  # None for full dataset
    "model_path": "model",
    "num_labels": 5,
    "pct_train": 0.8,
    "learning_rate": 0.001,
    "batch_size_train": 16,
    "batch_size_test": 1,
    "epochs": 2,
    "log_freq": 1,
}

color_map = {
    "date": (33, 33, 224),  # red
    "total": (0, 191, 0),  # green
    "address": (206, 18, 18),  # blue
    "company": (18, 103, 206),  # orange
}

task_1_dir = "0325updated.task1train(626p)"
