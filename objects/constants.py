config = {
    "tokenizer_name": "microsoft/layoutlm-base-uncased",
    "model_name": "microsoft/layoutlm-base-uncased",
    "data_path": "data",
    "n_samples": None,  # None for full dataset
    "model_path": "model",
    "num_labels": 5,
    "pct_train": 0.8,
    "learning_rate": 0.001,
    "batch_size_train": 4,
    "batch_size_test": 1,
    "epochs": 50,
    "log_freq": 5,
}

color_map = {
    0: (33, 33, 224),  # date, red
    1: (0, 191, 0),  # total, green
    2: (206, 18, 18),  # address, blue
    4: (18, 103, 206),  # company, orange
}

task_1_dir = "0325updated.task1train(626p)"
