import pandas as pd
import numpy as np
import sys
import torch                    
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from models import MultimodalFramework

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def main():
    """
    Runs unimodal models on test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - model_name (str): the name of the model to use (bert, resnet, or mlp - multilayer perceptron for metadata in Amazon reviews)
    - lr (str): the learning rate to use for the model
    - epochs (str): the number of epochs to train the model for
    - batch_size (int): the batch size to use for testing
    - random_seeds (list of int): a list of random seeds to use for testing
    - path (str): the path to the main data directory
    """
    args = sys.argv[1:]
    model_name = args[0]
    lr = args[1]
    epochs = args[2]
    batch_size = int(args[3])
    random_seeds = np.array(args[4].strip('][').split(', ')).astype(int) #a list of random seeds, for example: [42, 1, 67] 
    path = args[5] 
    df = pd.DataFrame(columns = ['accuracy', "precision", "recall", "f1-score", "CM", "CR"])

    for seed in random_seeds:
        model_path = path + 'unimodal_models_sentiment/best_model_'+str(lr)+'_' + str(seed)+'_adamW_' + str(epochs)+'_' + str(model_name)+ '.pth'
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        torch.cuda.empty_cache()

        model = MultimodalFramework(num_heads = 1, num_mod = 1)

        if model_name == "bert":
            test_inputs = torch.load(path + 'test_txt_inputs.pt')
            test_dataloader = DataLoader(test_inputs, batch_size=batch_size)

        elif model_name == "resnet":
            test_inputs = torch.load(path + 'test_img_inputs.pt')
            test_dataloader = DataLoader(test_inputs, batch_size=batch_size)
        else:
            test_inputs = torch.load(path + 'test_tab_inputs.pt')
            test_dataloader = DataLoader(test_inputs, batch_size=batch_size)

        model.load_state_dict(torch.load(model_path)) 
        model.to(device)

        correct = 0
        total = 0
        pred = []
        test_labels = []

        with torch.no_grad():
            for data in test_dataloader:
                if model_name == "bert":
                    inputs, masks, labels = data
                    inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                    outputs = model([inputs, masks], model_name)

                elif model_name == "resnet":
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs, model_name)
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    inputs, labels =inputs.float(), labels.long()
                    outputs = model(inputs, model_name)

                test_labels.extend(np.array(labels.cpu()))
                _, predicted = torch.max(outputs, 1)
                pred.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc= 100 * correct // total
        print(f'Accuracy: {100 * correct // total} %')

        test_labels = np.array(test_labels)
        cm = confusion_matrix(test_labels, pred)
        cr = classification_report(test_labels, pred, output_dict=True)
        df = df.append({'accuracy': acc, "precision":cr["macro avg"]["precision"]*100 ,
                        "recall":cr["macro avg"]["recall"]*100, "f1-score":cr["macro avg"]["f1-score"]*100,
                        "CM":cm, "CR":cr}, ignore_index=True)
        
    df.to_csv(model_name + "_results.csv")
    print(df.mean())
    
if __name__ == "__main__":
    main()
    
    
