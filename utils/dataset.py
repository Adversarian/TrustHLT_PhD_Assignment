from typing import List, Tuple

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def prepare_dataset(
    n_shadows: int = 10, tokenizer_id: str = "bert-base-uncased"
) -> Tuple[
    DatasetDict,
    DatasetDict,
    List[DatasetDict],
    PreTrainedTokenizerFast | PreTrainedTokenizer,
]:
    """Prepare the `yelp_review_full` dataset for use.

    Args:
        n_shadows (int, optional): Number of shadow models intended for the attack. Defaults to 10.
        tokenizer_id (str, optional): HuggingFace ID of the tokenizer to use. Defaults to "bert-base-uncased".

    Returns:
        Tuple[DatasetDict, DatasetDict, List[DatasetDict], PreTrainedTokenizerFast | PreTrainedTokenizer]:
        Original dataset, target subset and shadow subsets and the tokenizer object
    """
    dataset = load_dataset("yelp_review_full", split=["train+test"])[
        0
    ].train_test_split(test_size=0.5, stratify_by_column="label", seed=42)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        ),
        batched=True,
    )
    tokenized_dataset.set_format(type="torch")
    target_subset = tokenized_dataset["train"].train_test_split(
        test_size=0.5, stratify_by_column="label", seed=42
    )
    shadow_subsets = []
    for _ in range(n_shadows):
        shadow_subsets.append(
            tokenized_dataset["test"].train_test_split(
                test_size=0.5, stratify_by_column="label"
            )
        )
    return dataset, target_subset, shadow_subsets, tokenizer


def make_attack_dataset():
    pass
