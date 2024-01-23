# One-Versus-Others Multimodal Attention

This repository is the official implementation of [One-Versus-Others Multi-modal Attention](https://arxiv.org/abs/2030.12345). 

## Description
We present One-Versus-Others (OvO), a new scalable multimodal attention mechanism. The proposed formulation significantly reduces the computational complexity compared to the widely used early fusion through self-attention and cross-attention methods as it scales linearly with number of modalities and not quadratically. OvO outperformed self-attention, cross-attention, and concatenation on four diverse medical datasets, including four-modality, five-modality, and two six-modality datasets. The figure below demonstrated our model:

<img src="" width="300">

## Requirements
Python 3.9.0
PyTorch Version: 1.13.0+cu117
Torchvision Version: 0.14.0+cu117
To install requirements:

```setup
pip install -r requirements.txt
```
## Preprocessing
This paper uses four medical datasets (MIMIC, TADPOLE, TCGA, eICU), two non-medical datasets (Hateful Memes and Amazon Review) and one simulation dataset. The preprocessing steps for each dataset are located in their respective folder in the `README.md` files. The `common_files` folder contains scripts that are used by multiple datasets. 

## Training and hyperparameter tuning

The training in this paper is done hand in hand with hyperparameter tuning using Weights and Biases (Wandb). The training and tuning scripts follow a pattern `training_multimodal_hyper.py` or `training_unimodal_hyper.py`. So for example, to train a multimodal model using OvO attention on the MIMIC dataset, you would run the following command:

```train
python3 mimic/training_multimodal_hyper.py OvO /path/to/data /path/to/save/model /path/to/config wandb_project_title
```
An example config file is provided in `common_files/config.json`, which includes the full grid we used to find the best hyperparameters. Note that while non-medical datasets, Hateful Memes and Amazon reviews use similar pre-trained models such as Bert and ResNet, and MIMIC uses ClinicalBert, the other medical datasets and simulation dataset use regular neural network encoders. More details about exactly how to train each dataset are located in the `README.md` files inside each dataset folder.


## Evaluation

The evaluation scripts follow a pattern `evaluate.py` with a `multimodal` flag set to either True or False. For example, to evaluate a six modality model using OvO attention on the TADPOLE dataset, you would run the following command:

```evaluate
python3 tadpole/evaluate.py True OvO learning_rate epochs batch_size random_seed_list /path/to/test_data number_of_attention_heads
```
An example config file is provided in `common_files/config.json`, which includes the full grid we used to find the best hyperparameters. More details about exactly how to evaluate each dataset are located in the `README.md` files inside each dataset folder, as they differ slightly across datasets.


## Contributing Github Authors
[Michal Golovanevsky](https://github.com/michalg04), [Akira Nair](https://github.com/akira-nair), [Eva Schiller](https://github.com/eschill04)

