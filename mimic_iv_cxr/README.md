# MIMIC-IV + CXR dataset instructions
Pipeline to download the dataset, preprocess, train, and evaluate on the MIMIC-IV and CXR dataset.

### 1. Download data

To access the dataset, you must first be approved by PhysioNet.org for MIMIC-IV Notes (https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel) and MIMIC-CXR-JPG (https://physionet.org/content/mimic-cxr-jpg/2.0.0/). We use the same modalities (time-series and imaging) used in MedFuse and add two more by using demographic information and text notes from other MIMIC-IV datasets. You will need the admissions and patients tables from MIMIC-IV, and the discharge notes: https://physionet.org/content/mimic-iv-note/2.2/note/#files-panel. 

### 2. Preprocess and create PyTorch datasets
You must first run the preprocessing and test/train splitting that was done in MedFuse:
https://github.com/nyuad-cai/MedFuse/tree/main.
Following their preprocessing will generate the needed files used in our work for the imaging and time-series modalities. 
For the later stage preprocessing (e.g., tokenizing), we use our own functions. 
Finally, to pre-process the four modalities, run `python3 joining_with_text_dmeo.py /path_to_your/ehr_data_dir/ /path_to_your/cxr_data_dir /path_to_your/disharge.csv /path_to_your/core_dir /path_to_your/config phenotyping`. 

### 3. Training and hyperparameter tuning.

Unimodal:
To train and tune for a single modality model, use `training_unimodal_hyper.py`. This allows you to train a neural network for any one of the tabular modalities, and do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). 

Multi-modal:
To train on all four modalities, use `training_multimodal_hyper.py` which will do hyperparameter tuning using weights and biases (example config can be found in `common_files/config.json`). You can train using a concatenation model, a pairwise cross-modal attention model, an early fusion with self-attention model, and our OvO model. 

### 4. Evaluation

Unimodal:
To evaluate for a single modality model, use `evaluate.py` with the `multimodal` flag set to False. This allows you to evaluate the best model for any one of the modalities, using the best hyperparameters found in the training step. 

Multi-modal:
To evaluate on all six modalities, use `evaluate.py` with the `multimodal` flag set to True. This allows you to evaluate using the best hyperparameters found in the training step. 






