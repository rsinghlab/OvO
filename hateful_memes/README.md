# Hateful Memes Challenge dataset instructions
Pipeline to download the dataset, preprocess, train, and evaluate on the Hateful Memes Challenge dataset.

### 1. Download data, preprocess, and create PyTorch tensors.

Follow the instructions in  `creating_dataset.ipynb` to download the data. The notebook then create a train/test/val split, and turns the data into custom PyTorch datasets.


### 2. Training and hyperparameter tuning.

Unimodal:
Both Hateful Memes data and Amazon reviews data can be unimodally trained using `common_files/training_one_hyper.py`. This allows you to train a Ber or ResNet model for one modalitiy at a time, do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). 

Multi-modal:
To train using just two modalities, you can use `common_files/training_two_hyper.py`, for Bert and ResNet together. This will do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). You can train using a concatenation model, a pairwise cross-modal attention model, and our OvO model. 

### 3. Evaluation

Unimodal:
Both Hateful Memes data and Amazon reviews data can be unimodally evaluated using `common_files/evaluate_one.py`. This allows you to evaluate a Bert or ResNet model for one modalitiy at a time using the best hyperparameters found in the training step. 

Multi-modal:
To evluate using the two modalities, you can use `common_files/evaluate_two.py`, for Bert and ResNet together. This will evaluate using the best hyperparameters found in the training step. 
