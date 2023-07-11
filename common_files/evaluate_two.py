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
    Runs the multimodal framework on test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - model_name (str): the name of the model to use (e.g. bert_resnet, bert_resnet_OvO, bert_resnet_pairwise)
    - lr (str): the learning rate to use for the model
    - epochs (str): the number of epochs to train the model for
    - batch_size (int): the batch size to use for testing
    - num_heads (int): the number of attention heads to use in the model
    - random_seeds (list of int): a list of random seeds to use for testing
    - path (str): the path to the main data directory
    """

    args = sys.argv[1:]
    model_name = args[0] 
    lr = args[1]
    epochs = args[2]
    batch_size = int(args[3])
    num_heads = int(args[4])
    random_seeds = np.array(args[5].strip('][').split(', ')).astype(int) #a list of random seeds, for example: [42, 1, 67] 
    path = args[6] 
    df = pd.DataFrame(columns = ['accuracy', "precision", "recall", "f1-score", "CM", "CR"])

    
    for seed in random_seeds:
        model_path = path + 'two_modality_models/model_'+str(lr)+'_' + str(seed)+ "_" + str(epochs)+'_' + str(model_name)+ '.pth'
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        torch.cuda.empty_cache()

        model = MultimodalFramework(num_heads = num_heads, num_mod = 2)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        test_inputs_txt = torch.load(path + 'test_txt_inputs.pt')
        test_inputs_img = torch.load(path + 'test_img_inputs.pt')
        test_inputs_tab = torch.load(path + 'test_tab_inputs.pt')
        
        if model_name.split("_")[:2] == ["bert", "resnet"]:
            modality_1 = DataLoader(test_inputs_img, batch_size=batch_size)
            modality_2 = DataLoader(test_inputs_txt, batch_size=batch_size) 
            
        elif model_name.split("_")[:2] == ["resnet", "mlp"]:
            modality_1 = DataLoader(test_inputs_img, batch_size=batch_size) 
            modality_2 = DataLoader(test_inputs_tab, batch_size=batch_size)
            
        else:
            modality_1 = DataLoader(test_inputs_tab, batch_size=batch_size) 
            modality_2 = DataLoader(test_inputs_txt, batch_size=batch_size)
            

        correct = 0
        total = 0
        pred = []
        test_labels = []

        with torch.no_grad():
            for modality1, modality2 in zip(modality_1, modality_2):
                
                if model_name.split("_")[:2] == ["bert", "resnet"]:
                    text_inp, masks, text_labels = modality2
                    img_inp, labels = modality1

                    text_inp, masks, text_labels = text_inp.to(device), masks.to(device), text_labels.to(device)
                    img_inp, labels = img_inp.to(device), labels.to(device)

                    outputs = model([img_inp, text_inp, masks], model_name)

                elif model_name.split("_")[:2] == ["resnet", "mlp"]:
                    img_inp, labels = modality1
                    tab_inp, tab_labels = modality2
                    tab_inp, tab_labels = tab_inp.float(), tab_labels.long()

                    tab_inp, tab_labels = tab_inp.to(device), tab_labels.to(device)
                    img_inp, labels = img_inp.to(device), labels.to(device)

                    outputs = model([tab_inp, img_inp], model_name)
                else:
                    tab_inp, tab_labels = modality1
                    text_inp, masks, labels = modality2
                    tab_inp, tab_labels = tab_inp.float(), tab_labels.long()

                    tab_inp, tab_labels = tab_inp.to(device), tab_labels.to(device)
                    text_inp, masks, labels = text_inp.to(device), masks.to(device), labels.to(device)

                    outputs = model([tab_inp, text_inp, masks], model_name)

                test_labels.extend(np.array(labels.cpu()))
                _, predicted = torch.max(outputs, 1)
                pred.extend(predicted.cpu().numpy())
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc= 100 * correct // total
        print(f'Accuracy of the bert: {100 * correct // total} %')

        test_labels = np.array(test_labels)

        cm = confusion_matrix(test_labels, pred)
        cr = classification_report(test_labels, pred, output_dict=True)
        df = df.append({'accuracy': acc, "precision":cr["macro avg"]["precision"]*100 ,
                        "recall":cr["macro avg"]["recall"]*100, "f1-score":cr["macro avg"]["f1-score"]*100,
                        "CM":cm, "CR":cr}, ignore_index=True)
        
    df.to_csv(model_name + "_results.csv")
    print(df.mean())
    print(df.std())
    
if __name__ == "__main__":
    main()
    

    
    
