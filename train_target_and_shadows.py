from argparse import ArgumentParser

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from model.bert_configs import shadow, target
from model.train_configs import shadow_training_args, target_training_args
from utils.dataset import prepare_dataset
from utils.metrics import compute_acc
from utils.seed import seed_everything

seed_everything(42)


def main(
    args,
):
    ### PREPARE DATASETS
    _, target_subset, shadow_subsets, _ = prepare_dataset(
        n_shadows=args.n_shadows, tokenizer_id=args.tokenizer_id
    )

    ### TRAIN TARGET
    target_bert_config = BertConfig(**target)
    target_classifier = BertForSequenceClassification(config=target_bert_config)
    training_args = TrainingArguments(**target_training_args)
    target_trainer = Trainer(
        model=target_classifier,
        args=training_args,
        train_dataset=target_subset["train"],
        eval_dataset=target_subset["test"],
        compute_metrics=compute_acc,
    )
    print("Training Target model...")
    target_trainer.train()

    ### TRAIN SHADOWS
    cfg = shadow if not args.shadow_same_as_target else target
    shadow_bert_config = BertConfig(**cfg)
    for i in range(args.n_shadows):
        shadow_classifier = BertForSequenceClassification(config=shadow_bert_config)
        training_args = TrainingArguments(
            output_dir=f"saved_models/shadow_{i}", **shadow_training_args
        )
        shadow_trainer = Trainer(
            model=shadow_classifier,
            args=training_args,
            train_dataset=shadow_subsets[i]["train"],
            eval_dataset=shadow_subsets[i]["test"],
            compute_metrics=compute_acc,
        )
        print(f"Training Shadow model {i}...")
        shadow_trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training target and shadow models.")
    parser.add_argument(
        "-n",
        "--n_shadows",
        type=int,
        default=5,
        help="Number of shadow models to use. This must be consistent with during script runs. Defaults to `5`.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_id",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer to use for the bert model. Defaults to `bert-base-uncased`.",
    )
    parser.add_argument(
        "-sst",
        "--shadow_same_as_target",
        action="store_true",
        help="Set shadow model architecture to be exactly the same as target model.",
    )
    args = parser.parse_args()
    main(args)
