target_training_args = {
    "output_dir": "saved_models/target",
    "overwrite_output_dir": True,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "save_only_model": True,
    "seed": 42,
    "report_to": "none",
}

shadow_training_args = {
    "overwrite_output_dir": True,
    "num_train_epochs": 20,
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "save_only_model": True,
    "seed": 42,
    "report_to": "none",
}
