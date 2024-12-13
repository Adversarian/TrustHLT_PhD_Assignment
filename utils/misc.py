import os

from transformers.trainer_callback import TrainerState


# Taken from: https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
def ordinal(n: int) -> str:
    """Convert integer to ordinal string.

    Args:
        n (int): number

    Returns:
        str: ordinal string
    """
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    return str(n) + suffix


# Taken from: https://discuss.huggingface.co/t/loading-a-model-from-local-with-best-checkpoint/1707/9
def get_best_checkpoint_dir(trainer_save_dir: str) -> str | None:
    """Get best checkpoint according to eval metric from trainer directory.

    Args:
        trainer_save_dir (str): Directory to which huggingface trainer saved the models.

    Returns:
        str | None: Best checkpoint directory if it exists, None otherwise.
    """
    ckpt_dirs = os.listdir(trainer_save_dir)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpt_dirs[-1]
    state = TrainerState.load_from_json(
        f"{trainer_save_dir}/{last_ckpt}/trainer_state.json"
    )
    return state.best_model_checkpoint


def get_all_checkpoints(trainer_save_dir: str) -> list[int]:
    """Get all checkpoint numbers in a directory.

    Args:
        trainer_save_dir (str): Directory to which huggingface trainer saved the models.

    Returns:
        list[str]: Sorted list of checkpoint numbers.
    """
    ckpt_dirs = os.listdir(trainer_save_dir)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
    return [int(ckpt_dir.split("-")[1]) for ckpt_dir in ckpt_dirs]


def get_eval_history(trainer_save_dir: str) -> list[float]:
    """Get evaluation loss history for the given model.

    Args:
        trainer_save_dir (str): Directory to which huggingface trainer saved the models.

    Returns:
        list[float]: List of evaluation losses.
    """
    ckpt_dirs = os.listdir(trainer_save_dir)
    ckpt_dirs = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))
    last_ckpt = ckpt_dirs[-1]
    state = TrainerState.load_from_json(
        f"{trainer_save_dir}/{last_ckpt}/trainer_state.json"
    )
    return [
        entry["eval_loss"] for entry in state.log_history if "eval_loss" in entry.keys()
    ]
