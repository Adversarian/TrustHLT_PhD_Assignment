from argparse import ArgumentParser

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from xgboost import XGBClassifier

from utils.dataset import prepare_dataset
from utils.misc import get_best_checkpoint_dir
from utils.seed import seed_everything

seed_everything(42)


def main(args):
    ### TRAIN ATTACKER
    # NOTE: Because the same seed was set for dataset splitting
    # the shadow subsets will be the same across runs.
    _, _, shadow_subsets, _ = prepare_dataset(
        n_shadows=args.n_shadows, tokenizer_id=args.tokenizer_id
    )
    #### ATTACKER TRAIN DATASET CREATION
    try:
        saved_train_ds = np.load("saved_datasets/attack_train_dataset.npz")
        attack_X_train = saved_train_ds["attack_X_train"]
        attack_y_train = saved_train_ds["attack_y_train"]
        print("Cached dataset loaded.")
    except:
        print("Creating attack train dataset...")
        attack_dataset_X, attack_dataset_y = [], []
        for i in tqdm(range(args.n_shadows)):
            checkpoint_dir = get_best_checkpoint_dir(f"saved_models/shadow_{i}")
            shadow_model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint_dir
            ).cuda()
            shadow_train_dl = DataLoader(
                shadow_subsets[i]["train"],
                batch_size=256,
            )
            shadow_test_dl = DataLoader(
                shadow_subsets[i]["test"],
                batch_size=256,
            )
            shadow_model.eval()
            for batch in shadow_train_dl:
                with torch.inference_mode():
                    logits = shadow_model(
                        input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                    )["logits"]
                probs = F.softmax(logits, dim=-1)
                top_p, _ = probs.topk(k=args.k, dim=-1)
                attack_dataset_X.append(top_p.cpu().numpy())
                attack_dataset_y.append(np.ones((len(top_p), 1)))
            for batch in shadow_test_dl:
                with torch.inference_mode():
                    logits = shadow_model(
                        input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                    )["logits"]
                probs = F.softmax(logits, dim=-1)
                top_p, _ = probs.topk(k=args.k, dim=-1)
                attack_dataset_X.append(top_p.cpu().numpy())
                attack_dataset_y.append(np.zeros((len(top_p), 1)))
        attack_X_train = np.vstack(attack_dataset_X)
        attack_y_train = np.vstack(attack_dataset_y)
        np.savez_compressed(
            "saved_datasets/attack_train_dataset",
            attack_X_train=attack_X_train,
            attack_y_train=attack_y_train,
        )

    #### TRAINING
    base_xgb_attacker = XGBClassifier(objective="binary:logistic", eval_metric="auc")
    gridsearch_clf = GridSearchCV(
        base_xgb_attacker,
        {"max_depth": [1, 2, 3, 4, 5], "n_estimators": [5, 10, 20, 50]},
        verbose=1,
    )
    gridsearch_clf.fit(attack_X_train, attack_y_train)
    best_xgb_attacker = gridsearch_clf.best_estimator_
    best_xgb_attacker.save_model("saved_models/attacker/XGB_attacker.json")
    auc = roc_auc_score(
        attack_y_train, best_xgb_attacker.predict_proba(attack_X_train)[:, 1]
    )
    print(f"Attacker ROC-AUC on training data: {auc}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Script for training attacker model.")
    parser.add_argument(
        "-n",
        "--n_shadows",
        default=5,
        type=int,
        help="Number of shadow models to use. This must be consistent with during script runs. Defaults to `5`.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer_id",
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
