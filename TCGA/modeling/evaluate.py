import sys
sys.path.append('../../..')
import pandas as pd
import numpy as np
import argparse 
import torch                    
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from models import UnimodalFramework, MultimodalFramework
from common_files.custom_sets import TCGA_TabDataset, TCGA_ImgDataset, MyLoader

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def parse_bool(value):
    return value.lower() in ['true', '1', 'yes', 'y']

def parse_random_seeds(value):
    return np.array(value.strip('][').split(',')).astype(int)


def evaluate_unimodal(path, random_seeds, model_name, lr, batch_size, epochs):
    test_inputs = torch.load('tensor_data/' + str(model_name) +  '_test_inputs.pt')
    test_dataloader = DataLoader(test_inputs, batch_size=batch_size)
    
    numcols, x = next(iter(test_dataloader))
    model = UnimodalFramework(numcols.shape[1])
    
    df = pd.DataFrame(columns = ['accuracy', "precision", "recall", "f1-score", "CM", "CR"])
    
    for seed in random_seeds:
        model_path = path + '/baseline_models/model_' + str(lr) + "_" + str(seed) + "_" + str(epochs) + "_" + model_name + "_current.pth"
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        
        #if running on CPU and not CUDA, use: model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
        model.load_state_dict(torch.load(model_path)) 
        model.to(device)
        model.eval()

        correct = 0
        total = 0
        pred = []
        test_labels = []

        with torch.no_grad():
            for data in test_dataloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs, model_name)
 
                test_labels.extend(np.array(labels.cpu()))
                _, predicted = torch.max(outputs, 1)
                pred.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc= 100 * correct / total
        print(f'Accuracy: {acc} %')

        test_labels = np.array(test_labels)

        cm = confusion_matrix(test_labels, pred)
        cr = classification_report(test_labels, pred, output_dict=True)
        
        df = df.append({'accuracy': acc, "precision":cr["macro avg"]["precision"]*100 ,
                        "recall":cr["macro avg"]["recall"]*100, "f1-score":cr["macro avg"]["f1-score"]*100,
                        "CM":cm, "CR":cr}, ignore_index=True)
        
    df.to_csv(model_name + "_results.csv")
    print(df.mean())
    print(df.std())

def evaluate_multimodal(path, random_seeds, model_name, lr, batch_size, epochs, num_heads):
        modalities = ["clinical", "cnv", "epigenomic", "transcriptomic", "image"] #CSVs can be in any order, image should be last
        test_input_list = []
        modality_shapes = []

        for modality_name in modalities:
            test_inputs = torch.load(path + str(modality_name) +  '_test_inputs.pt')
            d, x = next(iter(test_inputs))
            #if not image
            if len(d.shape) != 3:
                modality_shapes.append(d.shape[0])

            test_input_list.append(DataLoader(test_inputs, batch_size=batch_size,shuffle=False))
    
        test_loader = MyLoader(test_input_list)
        model = MultimodalFramework(modality_shapes, num_heads)

        df = pd.DataFrame(columns = ['accuracy', "precision", "recall", "f1-score", "CM", "CR"])

        for seed in random_seeds:
            model_path = path + '/models/model_' + str(lr) + "_" + str(batch_size) + "_" \
            + str(seed) + "_" + str(epochs) + "_" + str(num_heads) + "_" + model_name + "_current.pth"

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(device)

            #if running on CPU and not CUDA, use: model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) 
            model.to(device)
            model.eval()

            correct = 0
            total = 0
            pred = []
            test_labels = []

            # since we're not training, we don't need to calculate the gradients for our outputs
            with torch.no_grad():
                for i, data in enumerate(test_loader):
                    mod1 = data[0]

                    inp1, labels = mod1
                    inp1, labels = inp1.to(device), labels.to(device)
                    inps = []
                    for i in range(len(data)):
                        inp, labels = data[i]
                        inp, labels = inp.to(device), labels.to(device)
                        inps.append(inp)

                    inp_len = labels.size(0)
                    outputs = model(inps, model_name)

                    test_labels.extend(np.array(labels.cpu()))
                    _, predicted = torch.max(outputs, 1)
                    pred.extend(predicted.cpu().numpy())
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc= 100 * correct / total
            print(f'Accuracy: {acc} %')

            test_labels = np.array(test_labels)

            cm = confusion_matrix(test_labels, pred)
            cr = classification_report(test_labels, pred, output_dict=True)

            df = df.append({'accuracy': acc, "precision":cr["macro avg"]["precision"]*100 ,
                            "recall":cr["macro avg"]["recall"]*100, "f1-score":cr["macro avg"]["f1-score"]*100,
                            "CM":cm, "CR":cr}, ignore_index=True)

        df.to_csv(modality_name + "_current_results.csv")
        print(df.mean())
        print(df.std())

def main():
    """
    Runs unimodal and multimodal models on test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - multimodal (bool): True or False, corresponds to evaluating a unimodal or multimodal model
    - model_name (str): the name of the modality if unimodal (e.g., demographics, aps, lab), or fusion type if multimodal (concatenation, OvO, self, cross)
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
    
    