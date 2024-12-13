# TrustHLT PhD Assignment
Entry for the assignment given by Prof. Dr. Ivan Habernal for the PhD application at TrustHLT (ANR 3979).

# Description
In this assignment, I was tasked to implement a simple Membership Inference Attack. Given the time constraints and the fact that the downstream task wasn't specified, this repository implements a MIA on sequence classification model. I specifically chose this task because MIA on auto-regressive language models is more involved and given my limited experience in attacking language models, I decided to stick with what I can readily comprehend and code in a time limit.

For the attack strategy, I have decided to implement the same strategy that was discussed in the [seminal paper that introduced MIA by Shokri et al.](https://arxiv.org/abs/1610.05820).

The general idea of the paper is to train several `shadow` models that an attacker hopes will mimic the behavior of the victim model. These shadow models will the be used to generate a dataset for an `attack` model based on output predictions or logits whose objective is to determine the membership of a given input sequence to the victim model's training data.

I will be using a BERT structure for both the target and shadow models with intentionally dissimilar internal structure to mimic a certain extent of black-box access for the attacker. The attacker in this scenario has a **grey-box** access in that he knows the general model architecture used for classification and he has access to data from a similar distribution to which the victim model was trained on but disjoint from the actual victim training data. However, access to weights, gradients and victim's training data, training regimen, and the exact specific internal architecture of the model(number of layers, number of attention heads, etc.) are **not** known to the attacker.

## Summary
- **Victim Downstream Task**: Sequence Intent Classification
- **Downstream Dataset**: [SNIPS](https://huggingface.co/datasets/benayas/snipsl)
- **MIA Method**: MIA through shadow models by [Shokri et al.](https://arxiv.org/abs/1610.05820)
- **Attacker Access Type**: Grey-box
- **Target Architecture**: BERT
- **Shadow Architecture**: BERT
- **Attack Model**: XGBoost with HP tuning



# Project Structure
- `model`: This directory houses the model definitions and training arguments for the proposed target and shadow models.
- `saved_datasets`: The created datasets for the attack model are saved here.
- `saved_models`: Pre-trained target and shadow models are saved here.
- `utils`: Contains utility functions required for dataset prepration, reproducability and metric calculations.
- `train_target_and_shadows.py`: This script is responsible for training the target and shadow models.
- `train_attacker.py`: This script is responsible for training the XGBoost attacker model.
- `inference.py`: This script tests the trained attacked against the target model on previously unseen data.
- `paper_review.pdf`: A copy of the review for the paper "ADePT: Auto-encoder based Differentially Private Text Transformation".

*All scripts have documentation where applicable*
# Usage
## 1. Installing the Project Requirements
```bash
$ pip install -r requirements.txt
```
## 2. Training Target and Shadow Models
This will save pre-trained model checkpoints to the `saved_models` directory. You may skip this step if you have previously pre-trained the models.
```bash
$ python train_target_and_shadow.py
```
The model architectures and training arguments can be found under `model/bert_configs.py` and `model/train_configs.py` respectively.

## 3. Training the Attacker Model
This will first create an attacker training dataset based on the data obtained from the shadow models and save it to `saved_datasets`. Then an XGBoost classifier will be fit on the training data and saved to `saved_models`.
```bash
$ python train_attacker.py
```
After the shadow models have been trained, we will once again go over the respective datasets for each shadow model and make a training dataset for the attacker model. For each shadow model, its training and test datasets will be fed to it to produce member and non-member classes for the attacker model respectively. After pooling the results from all shadow models, a gridsearch over the hyperparameters of the attacker model will be performed and the best performing attacker is saved to disk.
## 4. Run Experiment
This will test the trained attacker model against target model and reports the results.
```bash
$ python inference.py
```
Similar to how the training dataset for the attacker model was produced in the previous step, we will produce a test dataset but this time using the target model and its respective `train` and `test` dataset splits. It's important to note that neither the attacker model nor any of the shadow models have seen this data prior this step.

# Results
All 7 class probabilities from the shadow models were used to train the attacker. Below are their feature importances based on information gain. From this we can glean that the second probability contributed the most towards the decision of the attacker.
![Feature Importances](https://github.com/Adversarian/TrustHLT_PhD_Assignment/blob/main/figures/feature_importances.png)

Further more, the ROC-AUC of the attacker model and target model's evaluation loss were graphed against different checkpoints of the target model. The results can be seen below.
![AUC & Eval Loss vs. Checkpoint](https://github.com/Adversarian/TrustHLT_PhD_Assignment/blob/main/figures/roc_auc_and_eval_loss_vs_checkpoints.png)

This is surprising because based on the findings of Shokri et al., I expected generalization error (in this case "eval loss") to be correlated to attack success but this graph seems to imply otherwise. I suspect this might be due to an error in my implmenetation but I was unable to rectify it despite my best efforts given the time constraint.

The attacker model is much more successful on the training data generated by the shadow models, achieving a ROC-AUC of `0.9618`. (This can be verified through running `train_attacker.py`. This number might vary slightly due to random initialization of the models but I've done my best to ensure the reproducability of the results.)

# TODO
- ~~Investigate the effect of target's over/under-fitting by running inference against different target checkpoints.~~
- ~~Add more figures.~~
- ~~Move hardcoded arguments to an argument parser.~~
- ~~Test setting shadow architecture similar to target architecture.~~
- ~~Testing using many shadow models.~~
- Add more customization through command-line arguments. 