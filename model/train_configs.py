target_training_args = {
    "output_dir": "saved_models/target",
    "overwrite_output_dir": True,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "report_to": "none",
}

shadow_training_args = {
    "overwrite_output_dir": True,
    "num_train_epochs": 10,
    "per_device_train_batch_size": 64,
    "per_device_eval_batch_size": 64,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "report_to": "none",
}
