config = {
    "tokenizer_name": "microsoft/layoutlm-base-uncased",
    "model_name": "microsoft/layoutlm-base-uncased",
    "data_path": "data",
    "n_samples": 100,  # None for full dataset
    "model_path": "model",
    "num_labels": 5,
    "pct_train": 0.8,
    "learning_rate": 0.0001,
    "batch_size_train": 4,
    "batch_size_test": 1,
    "epochs": 2,
    "log_freq": 2,
}

color_map = {
    0: (33, 33, 224),  # red
    1: (0, 191, 0),  # green
    2: (206, 18, 18),  # blue
    4: (18, 103, 206),  # orange
}

task_1_dir = "0325updated.task1train(626p)"
