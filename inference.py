from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from xgboost import XGBClassifier

from utils.dataset import prepare_dataset
from utils.misc import get_all_checkpoints, get_eval_history, ordinal
from utils.seed import seed_everything

seed_everything(42)


def main(args):
    ### LOADING ATTACKER
    trained_xgb_attacker = XGBClassifier(objective="binary:logistic", eval_metric="auc")
    trained_xgb_attacker.load_model("saved_models/attacker/XGB_attacker.json")
    ### ATTACKER TEST DATASET CREATION
    # NOTE: Because the same seed was set for dataset splitting
    # the target subset will be the same across runs.
    _, target_subset, _, _ = prepare_dataset(tokenizer_id=args.tokenizer_id)
    checkpoints = get_all_checkpoints("saved_models/target")
    eval_history = get_eval_history("saved_models/target")
    roc_auc_scores = []
    #### ATTACKER TEST DATASET CREATION
    for checkpoint in tqdm(checkpoints):
        try:
            saved_test_ds = np.load(
                f"saved_datasets/attack_test_dataset_{checkpoint}.npz"
            )
            attack_X_test = saved_test_ds["attack_X_test"]
            attack_y_test = saved_test_ds["attack_y_test"]
            print(f"Cached dataset loaded for checkpoint {checkpoint}")
        except:
            print(f"Creating attack test dataset for checkpoint-{checkpoint}")
            attack_dataset_X, attack_dataset_y = [], []
            target_model = AutoModelForSequenceClassification.from_pretrained(
                f"saved_models/target/checkpoint-{checkpoint}"
            ).cuda()
            target_train_dl = DataLoader(
                target_subset["train"],
                batch_size=256,
            )
            target_test_dl = DataLoader(
                target_subset["test"],
                batch_size=256,
            )
            target_model.eval()
            for batch in target_train_dl:
                with torch.inference_mode():
                    logits = target_model(
                        input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                    )["logits"]
                probs = F.softmax(logits, dim=-1)
                top_p, _ = probs.topk(k=args.k, dim=-1)
                attack_dataset_X.append(top_p.cpu().numpy())
                attack_dataset_y.append(np.ones((len(top_p), 1)))
            for batch in target_test_dl:
                with torch.inference_mode():
                    logits = target_model(
                        input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                    )["logits"]
                probs = F.softmax(logits, dim=-1)
                top_p, _ = probs.topk(k=args.k, dim=-1)
                attack_dataset_X.append(top_p.cpu().numpy())
                attack_dataset_y.append(np.zeros((len(top_p), 1)))
            attack_X_test = np.vstack(attack_dataset_X)
            attack_y_test = np.vstack(attack_dataset_y)
            np.savez_compressed(
                f"saved_datasets/attack_test_dataset_{checkpoint}",
                attack_X_test=attack_X_test,
                attack_y_test=attack_y_test,
            )
        ### INFERENCE
        auc = roc_auc_score(
            attack_y_test, trained_xgb_attacker.predict_proba(attack_X_test)[:, 1]
        )
        print(f"Checkpoint: {checkpoint}, Attacker ROC-AUC on test data: {auc}")
        roc_auc_scores.append(auc)

    ### PLOT
    #### ROC AUC & EVAL LOSS vs. CHECKPOINT
    # plt.figure()
    # plt.plot(checkpoints, roc_auc_scores, marker="o", linestyle="-", color="b")
    # plt.title("ROC AUC Scores vs. Checkpoint")
    # plt.xlabel("Checkpoint")
    # plt.ylabel("ROC AUC Score")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("figures/roc_auc_vs_checkpoints.png")
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Checkpoint")
    ax1.set_ylabel("ROC AUC Score", color="tab:red")
    ax1.plot(checkpoints, roc_auc_scores, marker="o", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Eval Loss", color="tab:blue")
    ax2.plot(checkpoints, eval_history, marker="x", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    fig.tight_layout()
    plt.grid(True)
    plt.savefig("figures/roc_auc_and_eval_loss_vs_checkpoints.png")

    ### FEATURE IMPORTANCE
    feature_importances = trained_xgb_attacker.feature_importances_
    plt.figure()
    feature_names = [
        f"{ordinal(i+1)} highest prob" for i in range(len(feature_importances))
    ]
    plt.bar(feature_names, feature_importances, color="teal")
    plt.xticks(rotation=45)
    plt.title("Feature Importances")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/feature_importances.png")


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training attacker model.")
    parser.add_argument(
        "-t",
        "--tokenizer_id",
        type=str,
        default="bert-base-uncased",
        help="Tokenizer to use for the bert model. Defaults to `bert-base-uncased`.",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=7,
        help="`k` for top-k probabiltiies to use to build dataset. Defaults to `7`.",
    )
    args = parser.parse_args()
    main(args)
