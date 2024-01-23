import pandas as pd
import numpy as np
import sys
sys.path.append('../..')
import torch
import argparse                    
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from models import UnimodalFramework, MultimodalFramework
from data_utils import *
from common_files.custom_sets import *
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def parse_bool(value):
    return value.lower() in ['true', '1', 'yes', 'y']

def parse_random_seeds(value):
    return np.array(value.strip('][').split(',')).astype(int)

def evaluate_unimodal(path, random_seeds, model_name, lr, batch_size, epochs):
    ehr_data_dir = path
    
    task = "phenotyping"

    test_inputs = load_dataset(f'{ehr_data_dir}/{task}/{"test"}_{model_name}_dataset.pkl')
    test_dataloader = DataLoader(test_inputs, batch_size=batch_size)

    if model_name == "ts":
        numcols = next(iter(test_dataloader))[0].shape[2]
    else:
        numcols = next(iter(test_dataloader))[0].shape[1]

    model = UnimodalFramework(numcols)
    df = pd.DataFrame(columns=['accuracy', 'f1-score', 'AUROC', 'AUPRC'])

    for seed in random_seeds:
        model_path = 'models/model_pheno_' + str(lr) + "_" + str(seed) + "_" + str(batch_size) + "_" + str(epochs) + "_" + model_name + "_current.pth"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
        model.to(device)
        model.eval()

        total_preds = []
        total_labels = []

        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs, model_name)

                # Store all labels and predictions
                total_labels.extend(labels.cpu().numpy())
                total_preds.extend(outputs.sigmoid().cpu().numpy())

        total_labels = np.array(total_labels)
        total_preds = np.array(total_preds)

        # Calculate metrics
        accuracy = (total_preds.round() == total_labels).mean() * 100
        f1 = f1_score(total_labels, total_preds.round(), average='macro') * 100
        auroc = roc_auc_score(total_labels, total_preds, average='macro') * 100
        auprc = average_precision_score(total_labels, total_preds, average='macro') * 100

        df = df.append({'accuracy': accuracy, 'f1-score': f1, 'AUROC': auroc, 'AUPRC': auprc}, ignore_index=True)

    df.to_csv(model_name + "_current_results.csv")
    print(df.mean())
    print(df.std())

def evaluate_multimodal(path, random_seeds, model_name, lr, batch_size, epochs, num_heads):
    modalities = ["text", "img", "ts", "demo"]
    modality_shapes = {'text': 512, 'img': 3, 'ts': 29, 'demo': 10237}  # Replace with actual dimensions
    ehr_data_dir = path
    task = "phenotyping"

    # Data Loaders
    test_input_list = []
    for modality in modalities:
        test_inputs = load_dataset(f'{ehr_data_dir}/{task}/{"test"}_{modality}_dataset.pkl')
        test_input_list.append(DataLoader(test_inputs, batch_size=batch_size))
        
    test_loader = MyLoader(test_input_list)

    # Initialize model
    model = MultimodalFramework(modality_shapes, num_heads=num_heads)  # num_heads and other configs as per training

    metrics_df = pd.DataFrame(columns=['accuracy', 'f1-score', 'AUROC', 'AUPRC'])

    # Evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #model_5e-05_32_0_34_1_concat_current.pth
    for seed in random_seeds:
        model_path = 'models/model_' + str(lr) + "_" + str(batch_size) + "_" + str(seed) + "_" + str(epochs) + "_" + str(num_heads) + "_" + model_name + "_current.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        total_preds = []
        total_labels = []

        with torch.no_grad():
            
            for i, data in enumerate(test_loader):
                    
                # Extract modalities data
                text, img, ts, demo = data
                text_input, masks, text_labels = text
                img_input, img_labels = img
                ts_input, ts_labels = ts
                demo_input, demo_labels = demo
                # Move data to device
                text_input, masks, img_input, ts_input, demo_input, labels  = text_input.to(device), masks.to(device), img_input.to(device), ts_input.to(device), demo_input.to(device), text_labels.to(device)
                inp_len = labels.size(0)
                outputs = model([(text_input, masks), img_input, ts_input, demo_input], model_name)
                total_labels.extend(labels.cpu().numpy())
                total_preds.extend(outputs.sigmoid().cpu().numpy())

        total_labels = np.array(total_labels)
        total_preds = np.array(total_preds)

        # Calculate and store overall metrics
        accuracy = (total_preds.round() == total_labels).mean() * 100
        f1 = f1_score(total_labels, total_preds.round(), average='macro') * 100
        auroc = roc_auc_score(total_labels, total_preds, average='macro') * 100
        auprc = average_precision_score(total_labels, total_preds, average='macro') * 100
        metrics_df = metrics_df.append({'accuracy': accuracy, 'f1-score': f1, 'AUROC': auroc, 'AUPRC': auprc}, ignore_index=True)

    metrics_df.to_csv(model_name + "_metrics_results.csv")

    print("Metrics Summary:")
    print(metrics_df.mean())
    print(metrics_df.std())


def main():
    """
    Runs unimodal and multimodal models on test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - multimodal (bool): True or False, corresponds to evaluating a unimodal or multimodal model
    - model_name (str): the name of the modality if unimodal (e.g., demo, text), or fusion type if multimodal (concatenation, OvO, self, cross)
    - lr (str): the learning rate to use for the model
    - epochs (str): the number of epochs to train the model for
    - batch_size (int): the batch size to use for testing
    - random_seeds (list of int): a list of random seeds to use for testing
    - path (str): the path to the main data directory
    """
    parser = argparse.ArgumentParser(description='evaluation')

    # Add arguments
    parser.add_argument('multimodal', type=parse_bool, help='Multimodal flag (true/1/yes/y)')
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('lr', type=float, help='Learning rate')
    parser.add_argument('epochs', type=int, help='Number of epochs')
    parser.add_argument('batch_size', type=int, help='Batch size')
    parser.add_argument('random_seeds', type=parse_random_seeds, help='List of random seeds, e.g., [42, 1, 67]')
    parser.add_argument('path', type=str, help='Path to the dataset')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads (optional, default: 8)')


    # Parse the arguments
    args = parser.parse_args()

    # Accessing the arguments
    multimodal = args.multimodal
    model_name = args.model_name
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    random_seeds = args.random_seeds
    path = args.path
    num_heads = args.num_heads

    if not multimodal:
        evaluate_unimodal(path, random_seeds, model_name, lr, batch_size, epochs)
    else:
        evaluate_multimodal(path, random_seeds, model_name, lr, batch_size, epochs, num_heads)

    
if __name__ == "__main__":
    main()
    
    
