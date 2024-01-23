import pandas as pd
import numpy as np
import sys
import torch
import argparse                    
import torchvision
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from models import UnimodalFramework, MultimodalFramework
import sys
sys.path.append('../..')
from common_files.custom_sets import *
from metrics import *
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def parse_bool(value):
    return value.lower() in ['true', '1', 'yes', 'y']

def parse_random_seeds(value):
    return np.array(value.strip('][').split(',')).astype(int)


def evaluate_unimodal(path, random_seeds, model_name, lr, batch_size, epochs):

    test_inputs = torch.load((f'{path}/{model_name}_test_dataset.pt'))
    test_dataloader = DataLoader(test_inputs, batch_size=batch_size)

    numcols = next(iter(test_dataloader))[0].shape[1]

    model = UnimodalFramework(numcols)
    df = pd.DataFrame(columns=['accuracy', 'f1-score', 'MAUC', 'BCA'])

    for seed in random_seeds:
        model_path = path + '/models/model_' + str(lr) + "_" + str(seed) + "_" + str(batch_size) + "_" + str(epochs) + "_" + model_name + "_current.pth"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
        model.to(device)
        model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                # Convert outputs to probabilities
                probabilities = F.softmax(outputs, dim=1).detach().cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                labels_np = labels.detach().cpu().numpy()

                all_labels.append(labels_np)
                all_predictions.append(predictions)
                all_probabilities.extend(zip(labels_np, probabilities))

        # Concatenate all labels and predictions
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        # Calculate metrics
        accuracy = (all_predictions == all_labels.ravel()).mean() * 100
        f1 = f1_score(all_labels, all_predictions, average='macro') * 100
        num_classes = outputs.size(1)
        mauc = MAUC(all_probabilities, num_classes) * 100
        bca = calcBCA(all_predictions, all_labels, num_classes) * 100

        df = df.append({'accuracy': accuracy, 'f1-score': f1, 'MAUC': mauc, 'BCA': bca}, ignore_index=True)

    df.to_csv(model_name + "_current_results.csv")
    print(df.mean())
    print(df.std())

def evaluate_multimodal(path, random_seeds, model_name, lr, batch_size, epochs, num_heads):
    modalities = ["mri", "fdg_pet", "av45_pet", "csf", "cognitive_tests", "clinical"]    
    modality_shapes = {
        "mri": 662,
        "fdg_pet": 15,
        "av45_pet": 241,
        "csf": 9,
        "cognitive_tests": 15,
        "clinical": 24
    }

    # Data Loaders
    test_input_list = []
    for modality in modalities:
        test_inputs = torch.load((f'{path}/{modality}_test_dataset.pt'))
        test_input_list.append(DataLoader(test_inputs, batch_size=batch_size))
        
    test_loader = MyLoader(test_input_list)
    # Initialize model
    model = MultimodalFramework(modality_shapes, num_heads=num_heads)  # num_heads and other configs as per training

    metrics_df = pd.DataFrame(columns=['accuracy', 'f1-score', 'MAUC', 'BCA'])

    # Evaluation
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #model_5e-05_32_0_34_1_concat_current.pth
    for seed in random_seeds:
        model_path = path + '/models/model_' + str(lr) + "_" + str(batch_size) + "_" + str(seed) + "_" + str(epochs) + "_" + str(num_heads) + "_" + model_name + "_best.pth"
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            
            # Assume all loaders have the same length
            for i, data in enumerate(test_loader):
                    
                mri, fdg_pet, av45_pet, csf, cognitive_tests, clinical = data

                # Split into inputs and labels for each modality
                mri_input, mri_labels = mri
                fdg_pet_input, fdg_pet_labels = fdg_pet
                av45_pet_input, av45_pet_labels = av45_pet
                csf_input, csf_labels = csf
                cognitive_tests_input, cognitive_tests_labels = cognitive_tests
                clinical_input, clinical_labels = clinical

                # Move data to device
                mri_input, mri_labels = mri_input.to(device), mri_labels.to(device)
                fdg_pet_input, fdg_pet_labels = fdg_pet_input.to(device), fdg_pet_labels.to(device)
                av45_pet_input, av45_pet_labels = av45_pet_input.to(device), av45_pet_labels.to(device)
                csf_input, csf_labels = csf_input.to(device), csf_labels.to(device)
                cognitive_tests_input, cognitive_tests_labels = cognitive_tests_input.to(device), cognitive_tests_labels.to(device)
                clinical_input, clinical_labels = clinical_input.to(device), clinical_labels.to(device)

                inp_len = mri_labels.size(0)
                labels = mri_labels
                # Pass all modality inputs to model
                outputs = model([mri_input, fdg_pet_input, av45_pet_input, csf_input, cognitive_tests_input, clinical_input], model_name)

                # Convert outputs to probabilities
                probabilities = F.softmax(outputs, dim=1).detach().cpu().numpy()
                predictions = np.argmax(probabilities, axis=1)
                labels_np = labels.detach().cpu().numpy()

                all_labels.append(labels_np)
                all_predictions.append(predictions)
                all_probabilities.extend(zip(labels_np, probabilities))

        # Concatenate all labels and predictions
        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        # Calculate metrics
        accuracy = (all_predictions == all_labels.ravel()).mean() * 100
        f1 = f1_score(all_labels, all_predictions, average='macro') * 100
        num_classes = outputs.size(1)
        mauc = MAUC(all_probabilities, num_classes) * 100
        bca = calcBCA(all_predictions, all_labels, num_classes) * 100

        metrics_df = metrics_df.append({'accuracy': accuracy, 'f1-score': f1, 'MAUC': mauc, 'BCA': bca}, ignore_index=True)


    metrics_df.to_csv(model_name + "_metrics_results.csv")
    print("Metrics Summary:")
    print(metrics_df.mean())
    print(metrics_df.std())

def main():
    """
    Runs unimodal and multimodal models on test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - multimodal (bool): True or False, corresponds to evaluating a unimodal or multimodal model
    - model_name (str): the name of the modality if unimodal (e.g., mri, csf), or fusion type if multimodal (concatenation, OvO, self, cross)
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
    
    
