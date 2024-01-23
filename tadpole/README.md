# TADPOLE dataset instructions
Pipeline to download the dataset, preprocess, train, and evaluate on the TADPOLE dataset.

### 1. Download data
The challenge description is available at https://tadpole.grand-challenge.org/Data/. Follow the challenge's download instructions. Note: you must be approved for an ADNI account (https://adni.loni.usc.edu/data-samples/access-data/) to be able to access the data. The authors of this paper are not able to distribute any data relating to ADNI. 

### 2. Preprocess and create PyTorch datasets
Run `python3 find_columns.py ../path_to_your_data/TADPOLE_D1_D2.csv` to split the dataset into the available modalities. Then to pre-process the data (primarily imputing missing values), split the data into train/val/test, and create PyTorch datasets, follow the steps in  `create_torch_datasets.ipynb`. 

### 3. Training and hyperparameter tuning.

Unimodal:
To train and tune for a single modality model, use `training_unimodal_hyper.py`. This allows you to train a neural network for any one of the tabular modalities, and do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). 

Multi-modal:
To train on all six modalities, use `training_multimodal_hyper.py` which will do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). You can train using a concatenation model, a pairwise cross-modal attention model, an early fusion with self-attention model, and our OvO model. 

### 4. Evaluation

Unimodal:
To evaluate for a single modality model, use `evaluate.py` with the `multimodal` flag set to False. This allows you to evaluate the best neural network for any one of the modalities, using the best hyperparameters found in the training step. 

Multi-modal:
To evaluate on all six modalities, use `evaluate.py` with the `multimodal` flag set to True. This allows you to evaluate using the best hyperparameters found in the training step. 




