import sys
sys.path.append('../../..')
import pandas as pd
import numpy as np
import sys
import torch                    
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from models import  MultimodalFramework
from common_files.custom_sets import TCGA_TabDataset, TCGA_ImgDataset, MyLoader

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def main():
    """
    Runs the multimodal framework on test data, computes evaluation metrics, and saves results to a CSV file.
    Command-line arguments:
    - model_name (str): the name of the model to use (e.g. concat, pairwise, early, or OvO)
    - lr (str): the learning rate to use for the model
    - epochs (str): the number of epochs to train the model for
    - batch_size (int): the batch size to use for testing
    - heads (int): the number of attention heads to use in the model
    - random_seeds (list of int): a list of random seeds to use for testing
    - path (str): the path to the main data directory
    """

    args = sys.argv[1:]
    model_name = args[0] 
    lr = args[1]
    epochs = args[2]
    batch_size = int(args[3])
    heads = int(args[4])
    random_seeds = np.array(args[5].strip('][').split(', ')).astype(int) #a list of random seeds, for example: [42, 1, 67] 
    path = args[6] 
    
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
    model = MultimodalFramework(modality_shapes, heads)

    df = pd.DataFrame(columns = ['accuracy', "precision", "recall", "f1-score", "CM", "CR"])

    for seed in random_seeds:
        model_path = path + '/baseline_models/model_' + str(lr) + "_" + str(batch_size) + "_" \
        + str(seed) + "_" + str(epochs) + "_" + str(heads) + "_" + model_name + "_current.pth"

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
    
if __name__ == "__main__":
    main()
    
    
