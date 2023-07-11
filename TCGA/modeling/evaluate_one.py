import sys
sys.path.append('../../..')
import pandas as pd
import numpy as np
import sys
import torch                    
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from models import  UnimodalFramework
from common_files.custom_sets import TCGA_ImgDataset, TCGA_TabDataset

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def main():
    """
    Runs unimodal models on test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - modality_name (str): the name of the modality to use (e.g. "clinical", "cnv", "epigenomic", "transcriptomic", "image")
    - lr (str): the learning rate to use for the model
    - epochs (str): the number of epochs to train the model for
    - batch_size (int): the batch size to use for testing
    - random_seeds (list of int): a list of random seeds to use for testing
    - path (str): the path to the main data directory
    """
    args = sys.argv[1:]
    modality_name = args[0]
    lr = args[1]
    epochs = args[2]
    batch_size = int(args[3])
    random_seeds = np.array(args[4].strip('][').split(', ')).astype(int) 
    path = args[5]
    
    test_inputs = torch.load('tensor_data/' + str(modality_name) +  '_test_inputs.pt')
    test_dataloader = DataLoader(test_inputs, batch_size=batch_size)
    
    numcols, x = next(iter(test_dataloader))
    model = UnimodalFramework(numcols.shape[1])
    
    df = pd.DataFrame(columns = ['accuracy', "precision", "recall", "f1-score", "CM", "CR"])
    
    for seed in random_seeds:
        model_path = path + '/baseline_models/model_' + str(lr) + "_" + str(seed) + "_" + str(epochs) + "_" + modality_name + "_current.pth"
        
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
                
                outputs = model(inputs, modality_name)
 
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
        
    df.to_csv(modality_name + "_results.csv")
    print(df.mean())
    print(df.std())
    
if __name__ == "__main__":
    main()
    
    
