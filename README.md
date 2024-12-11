# TrustHLT PhD Assignment
Entry for the assignment given by Prof. Dr. Ivan Habernal for the PhD application at TrustHLT (ANR 3979).

# Description
In this assignment, I was tasked to implement a simple Membership Inference Attack. Given the time constraints and the fact that the downstream task wasn't specified, this repository implements a MIA on sequence classification model. I specifically chose this task because MIA on auto-regressive language models is more involved and given my limited experience in attacking language models, I decided to stick with what I can readily comprehend and code in a time limit.

For the attack strategy, I have decided to implement the same strategy that was discussed in the [seminal paper that introduced MIA by Shokri et al.](https://arxiv.org/abs/1610.05820).

The general idea of the paper is to train several `shadow` models that an attacker hopes will mimic the behavior of the victim model. These shadow models will the be used to generate a dataset for an `attack` model based on output predictions or logits whose objective is to determine the membership of a given input sequence to the victim model's training data.

I will be using a BERT structure for both the target and shadow models with intentionally dissimilar internal structure to mimic a certain extent of black-box access for the attacker. The attacker in this scenario has a **grey-box** access in that he knows the general model architecture used for classification and he has access to data from a similar distribution to which the victim model was trained on but disjoint from the actual victim training data. However, access to weights, gradients and victim's training data, training regimen, and the exact specific internal architecture of the model(number of layers, number of attention heads, etc.) are **not** known to the attacker.

## Summary
- **Victim Downstream Task**: Sequence Classification
- **Downstream Dataset**: [Yelp Reviews](https://huggingface.co/datasets/Yelp/yelp_review_full)
- **MIA Method**: MIA through shadow models by [Shokri et al.](https://arxiv.org/abs/1610.05820)
- **Attacker Access Type**: Grey-box
- **Target Architecture**: BERT
- **Shadow Architecture**: BERT
- **Attack Model**: XGBoost



# Project Structure
- `model`: This directory houses the model definitions and training arguments for the proposed target and shadow models.
- `saved_datasets`: The created datasets for the attack model are saved here.
- `saved_models`: Pre-trained target and shadow models are saved here.
- `utils`: Contains utility functions required for dataset prepration, reproducability and metric calculations.
- `experiment.ipynb`: Unedited prototype of my first attempt at this assignment which was later chopped up into individual scripts.
- `train_target_and_shadows.py`: This script is responsible for training the target and shadow models. If the pre-trained models.
- `train_attacker.py`: This script is responsible for training the XGBoost attacker model.
- `inference.py`: This script tests the trained attacked against the target model on previously unseen data.

*All scripts have documentation where applicable*
# Usage
## 1. Installing the Project Requirements
```bash
$ pip install -r requirements.txt
```
## 2. Training Target and Shadow Models
This will save pre-trained model checkpoints to the `saved_models` directory. You may skip this step if you have previously pre-trained the models.
```bash
$ python3 train_target_and_shadow.py
```
## 3. Training the Attacker Model
This will first create an attacker training dataset based on the data obtained from the shadow models and save it to `saved_datasets`. Then an XGBoost classifier will be fit on the training data and saved to `saved_models`.
```bash
$ python3 train_attacker.py
```

## 4. Run Experiment
This will test the trained attacker model against target model and reports the results.
```bash
$ python3 inference.py
```

# Results
TODO.

# TODO
- Investigate the effect of target's over/under-fitting by running inference against different target checkpoints. 