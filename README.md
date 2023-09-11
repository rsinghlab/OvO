# One-Versus-Others Multi-modal Attention

This repository is the official implementation of [One-Versus-Others Multi-modal Attention](https://arxiv.org/abs/2030.12345). 

## Description
We present a multi-modal attention method that is domain-agnostic and does not rely on modality alignment or pairwise calculations, called as one-versus-others (OvO) attention. OvO offers a novel approach that takes outputs from one modality encoder and computes the dot product against a weight matrix and the average of the weights from all other modalities encoders (hence the name, one versus others); this is repeated for each modality. Our approach reduces the computational complexity significantly, growing liniearly with respect to number of modalities, unlike early fusion and cross-modal architecures which grow quadratically. The figure below demonstrated our model:

<img src="" width="300">

## Requirements
Python 3.7.4 (and above)
To install requirements:

```setup
pip install -r requirements.txt
```
## Preprocessing
This paper uses three real-world datasets (Hateful Memes, Amazon Reviews, and TCGA), and one simulation dataset. The preprocessing steps for each dataset are located in their respective folder in the `README.md` files. The `common_files` folder contains scripts that are used by multiple datasets. 

## Training and hyperparameter tuning

The training in this paper is done hand in hand with hyperparameter tuning using Weights and Biases (Wandb). The training and tuning scripts follow a pattern `training_NUMBER-OF-MODALITIES_hyper.py`. So for example, to train a three modality model using OvO attention on the Amazon dataset, you would run the following command:

```train
python3 amazon_reviews/training_three_hyper.py bert_resnet_mlp_OvO /path/to/data /path/to/save/model /path/to/config wandb_project_title
```
An example config file is provided in `common_files/config.json`, which includes the full grid we used to find the best hyperparameters. Note that while Hateful Memes and Amazon reviews use similar pre-trained models such as Bert and ResNet, the TCGA dataset and simulation dataset use regular neural network encoders and have slightly different training scripts. More details about exactly how to train each dataset are located in the `README.md` files inside each dataset folder.


## Evaluation

The evaluation scripts follow a pattern `evaluate_NUMBER-OF-MODALITIES.py`. For example, to evaluate a three modality model using OvO attention on the Amazon dataset, you would run the following command:

```train
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

Note that the Hateful Memes Challenge was closed by the time we wrote the paper, so we did not have access to the true test labels used in the competition. Thus, these results cannot be directly compared to other competiton participants, but should still provide a good estimation. 

## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
