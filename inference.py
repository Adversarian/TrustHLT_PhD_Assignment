import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from xgboost import XGBClassifier

from utils.dataset import prepare_dataset
from utils.seed import seed_everything

seed_everything(42)


def main(args={"checkpoints": [27350]}):
    ### LOADING ATTACKER
    trained_xgb_attacker = XGBClassifier(objective="binary:logistic", eval_metric="auc")
    trained_xgb_attacker.load_model("saved_models/attacker/XGB_attacker.json")
    ### ATTACKER TEST DATASET CREATION
    # NOTE: Because the same seed was set for dataset splitting
    # the target subset will be the same across runs.
    _, target_subset, _, _ = prepare_dataset()
    #### ATTACKER TEST DATASET CREATION
    for checkpoint in sorted(args["checkpoints"]):
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
            target_train_dl = DataLoader(target_subset["train"], batch_size=100)
            target_test_dl = DataLoader(target_subset["test"], batch_size=100)
            target_model.eval()
            for batch in target_train_dl:
                with torch.inference_mode():
                    logits = target_model(
                        input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                    )["logits"]
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                attack_dataset_X.append(probs)
                attack_dataset_y.append(np.ones((len(probs), 1)))
            for batch in target_test_dl:
                with torch.inference_mode():
                    logits = target_model(
                        input_ids=batch["input_ids"].cuda(),
                        attention_mask=batch["attention_mask"].cuda(),
                    )["logits"]
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                attack_dataset_X.append(probs)
                attack_dataset_y.append(np.zeros((len(probs), 1)))

            attack_X_test = np.vstack(attack_dataset_X).sort(axis=-1)
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


if __name__ == "__main__":
    main()
