# Amazon reviews datasets instructions
Pipeline to download the dataset, preprocess, train, and evaluate on the Amazon reviews dataset.

### 1. Download data

Download the data from https://nijianmo.github.io/amazon/ by clicking on the "Electronics" section. Click on both the electornics reviews and electronics metadata.
Run `python3 download_image.py ../path_to_your_data/amazon/Electronics/` to download the image data. 

### 2. Preprocess and create PyTorch datasets
Run `python3 create_datasets.py ../path_to_your_data/amazon/Electronics/` to create a tabular, a text, and an imaging PyTorch datasets used for modeling for each train/test/val. 

### 3. Training and hyperparameter tuning.

Unimodal:
Both Hateful Memes data and Amazon reviews data can be unimodally trained using `common_files/training_one_hyper.py`. This allows you to train a Bert, ResNet, or an multi-layer perceptron model for one modalitiy at a time, do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). 

Multi-modal:
To train using just two modalities, you can use `common_files/training_two_hyper.py`, for Bert and ResNet together. To train on all three modalities, use `training_three_hyper.py` which will do hyperparameter tuning using Weights and Biases (example config can be found in `common_files/config.json`). You can train using a concatenation model, a pairwise cross-modal attention model, an early-fusion with self-attention model, and our OvO model. 

### 4. Evaluation

Unimodal:
Both Hateful Memes data and Amazon reviews data can be unimodally evaluated using `common_files/evaluate_one.py`. This allows you to evaluate a Bert, ResNet, or an multi-layer perceptron model for one modalitiy at a time using the best hyperparameters found in the training step. 

Multi-modal:
To evluate using just two modalities, you can use `common_files/evaluate_two.py`, for Bert and ResNet together. To evaluate on all three modalities, use `evaluate_three.py` which will evaluate using the best hyperparameters found in the training step. 




