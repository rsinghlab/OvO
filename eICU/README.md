# eICU dataset instructions
Pipeline to download the dataset, preprocess, train, and evaluate on the eICU dataset.

### 1. Download data

Download the data from https://physionet.org/content/eicu-crd/2.0/. You must be approved by PhysioNet.org first. For more information on the dataset, visit: https://eicu-crd.mit.edu/about/eicu/ . 

### 2. Preprocess and create PyTorch datasets
To preprocess the eICU data, begin by unpacking the compressed files from the downloaded tables of interest. Next, open `preprocessing.ipynb`, which contains the preprocessing pipeline to generate .csv files for each of the modalities. Make sure to specify the filepaths to each of the unpacked csv files and the output paths to save preprocessed files. After running that notebook, run `train_test_split.ipynb` to concatenate modalities using patient unit stay ID and split data into training, validation, and testing subsets on a patient ID level. You may also adjust the random seed to create different splits. Finally, to create PyTorch datasetes for each modality, run to `creating_torch_datasets.ipynb`. 

### 3. Training and hyperparameter tuning.

Unimodal:
To train and tune for a single modality model, use `training_unimodal_hyper.py`. This allows you to train a neural network for any one of the tabular modalities, and do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). 

Multi-modal:
To train on all six modalities, use `training_multimodal_hyper.py` which will do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). You can train using a concatenation model, a pairwise cross-modal attention model, an early fusion with self-attention model, and our OvO model. 

### 4. Evaluation

Unimodal:
To evaluate for a single modality model, use `evaluate.py` with the `multimodal` flag set to False. This allows you to evaluate the best neural network for any one of the tabular modalities, using the best hyperparameters found in the training step. 

Multi-modal:
To evaluate on all six modalities, use `evaluate.py` with the `multimodal` flag set to True. This allows you to evaluate using the best hyperparameters found in the training step. 




