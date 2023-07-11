# Preprocessing for TCGA Datasets (from data_by_cases)
Pipeline to permanently assign Case IDs to training, testing, and validation categories, then move the raw data from a by-case format to preprocessed tensors split via the pre-assigned IDs. 

### 1. Preprocess the non-image modalities

Run `create_<modality>_dataset.ipynb` for all five modalities. This will organize the data in `.csv` form (or in an image folder) and perform some preliminary data cleaning such as dropping columns with NaN values and selecting highly variable features.

### 2. Find the intersection of patients who have all five modalities, and then assign Case IDs to train, test, and validate

Follow the steps in `combine_modalities.ipynb` with the appropriate data paths. This notebook will filter case IDs that exist for all five modalities. Then, the case IDs are combined and randomly assigned to train, test, and validation sets. These assignments are  then saved them to a `.csv` file at your specified path.

### 3. Feature reduction and Pytorch tensor conversion

Follow the steps in `reduce_features_create_tensors` with the appropriate data paths. This notebook creates three possible datasets for each of the gene expression (transcriptomic), CNV, and DNA Methylation (epigenomic) datasets, based on 50, 100, 150 RandomForest estimators as a way to reduce features. These datasets can be used when models are tuned to pick which preforms the best (only on validation data, not test data). Then, every dataset is converted to a pytorch tensor custom dataset and saved as `.pt` files.





