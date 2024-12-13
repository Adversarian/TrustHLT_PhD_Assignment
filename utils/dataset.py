from typing import List, Tuple

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from model.bert_configs import target


def prepare_dataset(
    n_shadows: int = 5, tokenizer_id: str = "bert-base-uncased", seed=42
) -> Tuple[
    DatasetDict,
    DatasetDict,
    List[DatasetDict],
    PreTrainedTokenizerFast | PreTrainedTokenizer,
]:
    """Prepare the `benayas/snips` dataset for use. Splits the dataset into target and shadow subsets.
    The dataset is first split into two halves. The first half will be solely used by the target model
    for training and evalution while the second half is available for theshadow models. This way we
    sure that the shadow models have access to mutually exclusive data with regard to the target.

    Args:
        n_shadows (int, optional): Number of shadow models intended for the attack. Defaults to 10.
        tokenizer_id (str, optional): HuggingFace ID of the tokenizer to use. Defaults to "bert-base-uncased".

    Returns:
        Tuple[DatasetDict, DatasetDict, List[DatasetDict], PreTrainedTokenizerFast | PreTrainedTokenizer]:
        Original dataset, target subset and shadow subsets, and the tokenizer object.
    """
    dataset = load_dataset("benayas/snips", split=["train+test"])[0]
    dataset = dataset.class_encode_column("category")
    dataset = dataset.rename_column("category", "label")
    dataset = dataset.train_test_split(
        test_size=0.5, stratify_by_column="label", seed=seed
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x["text"],
            return_tensors="pt",
            padding="max_length",
            max_length=target["max_position_embeddings"],
            truncation=True,
        ),
        batched=True,
    )
    tokenized_dataset.set_format(type="torch")
    target_subset = tokenized_dataset["train"].train_test_split(
        test_size=0.5, stratify_by_column="label", seed=seed
    )
    shadow_subsets = []
    for _ in range(n_shadows):
        shadow_subsets.append(
            tokenized_dataset["test"].train_test_split(
                test_size=max(1 / (n_shadows + 1), 0.2),
                train_size=max(1 / (n_shadows + 1), 0.2),
                stratify_by_column="label",
                seed=seed,
            )
        )
    return dataset, target_subset, shadow_subsets, tokenizer
