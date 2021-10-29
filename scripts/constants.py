config = {
    "tokenizer_name": "microsoft/layoutlm-base-uncased",
    "model_name": "microsoft/layoutlm-base-uncased",
    "data_path": "data",
    "model_path": "model",
    "num_labels": 5,
    "pct_train": 0.8,
    "learning_rate": 0.0001,
    "batch_size_train": 4,
    "batch_size_test": 1,
    "epochs": 4,
    "log_freq": 2,
}

color_map = {
    0: "red",
    1: "green",
    2: "blue",
    4: "orange",
}
