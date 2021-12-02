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
    "epochs": 100,
    "log_freq": 2,
}

task_1_dir = "0325updated.task1train(626p)"
