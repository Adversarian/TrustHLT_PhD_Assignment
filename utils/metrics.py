from typing import Dict, Tuple

import evaluate
import numpy as np


def compute_acc(eval_preds: Tuple[np.ndarray, np.ndarray]) -> Dict:
    """compute metrics function for the trainer

    Args:
        eval_preds (Tuple[np.ndarray, np.ndarray]): evaluation predictions

    Returns:
        Dict: accuracy
    """
    accuracy = evaluate.load("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)
