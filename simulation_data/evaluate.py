import sys
sys.path.append('../..')
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from common_file.model_utils import MyLoader
from models import OvO, pairwise, concat, early

def eval(model, test_loader):
    """
    Evaluate the performance of the given multimodal model on the simulation test set.

    Parameters:
    - model: torch.nn.Module
        The PyTorch model to be evaluated.
    - test_loader: torch.utils.data.DataLoader
        The data loader for the test dataset.

    Returns:
    - acc: float
        The accuracy of the model on the test dataset.
    - pred: list of ints
        The predicted labels for the test dataset.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    pred = []
    test_labels = []
    print("evaluating")
    
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
            outputs = model(inps)
            
            test_labels.extend(np.array(labels.cpu()))
            _, predicted = torch.max(outputs, 1)
            pred.extend(predicted.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc= 100 * correct / total
    print(f'Accuracy: {100 * correct / total} %')
    return acc, pred


def main():
    """
    Runs multi-modal models on simulation test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - model_name (str): the name of the model to use (concat, OvO, early, or pairwise)
    - lr (str): the learning rate to use for the model
    - epochs (str): the number of epochs to train the model 
    - heads (int): the number of attention heads to use in the model
    - batch_size (int): the batch size to use for testing
    - random_seeds (list of int): a list of random seeds to use for testing
    - path (str): the path to the main data directory
    """
    args = sys.argv[1:]
    model_name = args[0] 
    lr = args[1]
    epochs = args[2]
    heads = int(args[3])
    batch_size = int(args[4])
    random_seeds = np.array(args[5].strip('][').split(', ')).astype(int) 
    path_to_data = args[6] 
    
    df = pd.DataFrame(columns = ["num_modalities", "model_name",  "test_accuracy", "test_list", "sd"]) #,"FLOPS" 
    num_modalities = [2,5,10,15,20] #we tuned on 2, 5, 10, 15, and 20 simulated modalities, but any >= 2 would work
    for i in num_modalities: 
        if model_name == "concat":
            model = concat(i)
        elif model_name == "pairwise":
            model = pairwise(i,heads)
        elif model_name == "early":
            model = early(i,heads)
        else:
            model = OvO(i,heads)            
        acc = 0
        acc_list = []
        preds = []
        for seed in random_seeds: 
            print(seed)
            model_path = path_to_data + '/models/model_' + str(lr) + "_" + str(batch_size) + "_" + str(seed) + "_" + str(epochs)+ "_" + str(heads) + "_" + model_name + "_" + str(i)+ '_current.pth'
            model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu'))) 
            test_input_list = []

            for j in range(i):
                test_inputs = torch.load(path_to_data + "test_modality_" + str(j) +  "_inputs.pt")
                test_input_list.append(DataLoader(test_inputs, batch_size=batch_size, shuffle=False))
            test_loader = MyLoader(test_input_list) 
            
            a, pred = eval(model, test_loader) 
            acc += a
            acc_list.append(a)
            preds.append(pred)

        df = df.append({'num_modalities': i, "model_name": model_name, "test_accuracy":acc/10, "test_list":acc_list, "sd": np.array(acc_list).std()}, ignore_index=True) #"FLOPS": flops
        df.to_csv(model_name + "_HP_results.csv", index=False)

    
if __name__ == "__main__":
    main()