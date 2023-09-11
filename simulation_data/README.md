# Simulation dataset instructions
Pipeline to create the simulation dataset, train, and evaluate.

### 1. Create data, preprocess, and save PyTorch tensors.

Follow the instructions in `create_simulation_data.ipynb` to create the data as described in the paper. The notebook then create a train/test/val split, and turns the data into PyTorch datasets.


### 2. Training and hyperparameter tuning.

For the experiment in the paper, we trained and tuned 2, 5, 10, 15, and 20 simulated modalities together, using `training_hyper.py`. However, any number of modalities >= 2 would work. This script will do hyperparameter tuning using Weights and Biases (example config can be found in `common_files/config.json`). You can train using a concatenation model, a pairwise cross-modal attention model, early-fusion with self-attention model, and our OvO model. 

### 3. Evaluation

To evluate using the simulated modalities, you can use `evaluate.py`. This will evaluate using the best hyperparameters found in the training step. 
