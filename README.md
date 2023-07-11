# One-Versus-Others Multi-modal Attention

This repository is the official implementation of [One-Versus-Others Attention: Scalable Multimodal
Integration](https://arxiv.org/). 

## Description
We present a multi-modal attention method that is domain-agnostic and does not rely on modality alignment or pairwise calculations, called as one-versus-others (OvO) attention. OvO offers a novel approach that takes outputs from one modality encoder and computes the dot product against a weight matrix and the average of the weights from all other modalities encoders (hence the name, one versus others); this is repeated for each modality. Our approach reduces the computational complexity significantly, requiring only n computations instead of n choose 2 (cross-modal attention), where n represents the number of modalities. The figure below demonstrates our model:

<img src="https://user-images.githubusercontent.com/35315239/234658728-aac1979a-fc3b-4bf8-9f88-877e6b8593c2.png" width="600">

## Requirements
Python 3.7.4 (and above)

To install requirements:

```setup
pip3 install -r requirements.txt
```
## Preprocessing
This paper uses three real-world datasets (Hateful Memes, Amazon Reviews, and TCGA) and one simulation dataset. The preprocessing steps for each dataset are located in their respective folder in the `README.md` files. The `common_files` folder contains scripts that are used by multiple datasets. 

## Training and hyperparameter tuning

The training in this paper is done hand in hand with hyperparameter tuning using weights and biases. The training and tuning scripts follow a pattern `training_NUMBER-OF-MODALITIES_hyper.py`. For example, to train a three modality model using OvO attention on the Amazon dataset, you would run the following command:

```train
python3 amazon_reviews/training_three_hyper.py bert_resnet_mlp_OvO /path/to/data /path/to/save/model /path/to/config wandb_project_title
```
An example config file is provided in `common_files/config.json`, which includes the full grid we used to find the best hyperparameters. Note that while Hateful Memes and Amazon reviews use similar pre-trained models such as Bert and ResNet, the TCGA dataset and simulation dataset use regular neural network encoders and have slightly different training scripts. More details about exactly how to train each dataset are located in the `README.md` files inside each dataset folder.


## Evaluation

The evaluation scripts follow the pattern `evaluate_NUMBER-OF-MODALITIES.py`. For example, to evaluate a three modality model using OvO attention on the Amazon dataset, you would run the following command:

```evaluate
python3 amazon_reviews/evaluate_three.py bert_resnet_mlp_OvO learning_rate epochs batch_size number_of_attention_heads random_seed_list /path/to/test_data
```
An example config file is provided in `common_files/config.json`, which includes the full grid we used to find the best hyperparameters. More details about exactly how to evaluate each dataset are located in the `README.md` files inside each dataset folder, as they differ slightly across datasets.


## Results

Our model achieves the following performance on the three datasets:

| OvO model on:      | Accuracy (%)    | F1-Score (%)   |
| ------------------ |---------------- | -------------- |
| Hateful Memes      |  70.7 +- 0.87   | 67.7 +- 0.97   |
| ------------------ |---------------- | -------------- |
| Amazon Reviews     |  93.1 +- 0.30   | 93.0 +- 0.31   |
| ------------------ |---------------- | -------------- |
| TCGA               |  98.3 +- 0.07   | 98.4 +- 0.92   |
| ------------------ |---------------- | -------------- |

Note that the Hateful Memes Challenge was closed by the time we wrote the paper, so we did not have access to the true test labels used in the competition. Thus, these results cannot be directly compared to other competition participants but should still provide a good estimation. 

