import numpy as np
import json
import sys
sys.path.append('../..')
import logging
import torch                    
import torch.nn as nn
import torchvision
import time
import pprint
from sklearn.metrics import f1_score,roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import wandb
from models import UnimodalFramework
from common_files.model_utils import set_seed, build_optimizer
from common_files.custom_sets import *
from data_utils import *
import torch.nn.functional as F

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

def train_model(modality_name, dataloaders, criterion, config, path):
    """
    Trains a unimodal deep learning model using the given data loaders, criterion, and optimizer, and wandb config. Logs metrics to wandb (weights and biases) for hyperparameter tuning.

    Args:
        modality_name (str): The name of the unimodal modality to be trained (e.g., clinica, mri).
        dataloaders (Dict[str, DataLoader]): A dictionary containing PyTorch DataLoader objects for the 'train' and 'val' sets.
        criterion (Callable): The loss function to optimize during training.
        config: wandb config with hyperparameter values.
        path (str): The file path where the trained model will be saved.

    Returns:
        model (nn.Module): The trained unimodal model.
        history (Dict[str, List[float]]): A dictionary containing training and validation accuracy and loss history.
    """
    set_seed(config.random_seed)
    
    if modality_name == "ts":
        numcols = next(iter(dataloaders["train"]))[0].shape[2]
    else:
        numcols = next(iter(dataloaders["train"]))[0].shape[1]
        
    model = UnimodalFramework(numcols) 
    
    num_epochs = config.epochs
    

    optimizer = build_optimizer(model, config.optimizer,config.learning_rate, 0.9)

    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []


    best_acc = 0.0
    patience = 10 #patience and trigger is to stop the run if the validaiton accuracy is not imporving
    trigger = 0
    acc_dict = {}
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            preds_list = []
            labels_list =[]


            running_loss = 0.0
            running_corrects = 0
            preds_list = []
            labels_list =[]


            for data in dataloaders[phase]:
                torch.cuda.empty_cache()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                optimizer.zero_grad()

              
                with torch.set_grad_enabled(phase == 'train'):

                    model.to(device)
                    if modality_name == "text":
                        inputs, masks, labels = data
                        inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                        outputs = model([inputs, masks], modality_name) #, modality_name
                    else:
                        inputs, labels = data
                        inputs,labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs, modality_name)
                    
                    loss = criterion(outputs, labels.float())
                    preds = outputs.sigmoid() > 0.5

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                labels_list.extend(labels.cpu())
                preds_list.extend(preds.cpu())
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            #epoch_f1 = classification_report(np.array(labels_list), np.array(preds_list), output_dict=True)["macro avg"]["f1-score"]
            epoch_f1 = f1_score(labels_list, preds_list, average='macro')
            epoch_auroc = roc_auc_score(labels_list, preds_list, average='macro')
            epoch_auprc = average_precision_score(labels_list, preds_list, average='macro')

            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc,  "val_f1": epoch_f1, "val_auroc": epoch_auroc, "val_auprc": epoch_auprc}) #this logs to weights and biases
               
                acc_dict[epoch] = float(epoch_f1)
                val_acc_history.append(epoch_f1)
                val_loss_history.append(epoch_loss)
                
                #saving the model at its current state
                torch.save(model.state_dict(), path + "_current.pth")
                if epoch_f1 > best_acc:
                    best_acc = epoch_f1
                    
                    #saving the model at its best validation accuracy
                    torch.save(model.state_dict(), path + "_best.pth")
                
                #this code chunk is to stop the run if it doesn't progress for 10 epochs 
                
                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger +=1
                    if trigger >= patience:
                        return model, {"train_acc":train_acc_history, "val_acc":val_acc_history,"train_loss":train_loss_history, "val_loss":val_loss_history}
                else:
                    trigger = 0
                 
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc,"train_f1": epoch_f1, "train_auroc": epoch_auroc, "train_auprc": epoch_auprc, "epoch": epoch}) #this logs to weights and biases
                
                train_acc_history.append(epoch_acc.detach().cpu())
                train_loss_history.append(epoch_loss)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    
    return model, {"train_acc":train_acc_history, "val_acc":val_acc_history,"train_loss":train_loss_history, "val_loss":val_loss_history}

         
        
def main():
    """
    Trains a unimodal PyTorch model with Weights and Biases (wandb) and hyperparameter tuning.

    Arguments:
        modality_name (str): The name of the modality to train.
        data_path (str): The path to the directory containing the training and validation data.
        save_model_path (str): The path to the directory where the trained model will be saved.
        config_path (str): The path to the directory containing the configuration file for the hyperparameter search.
        wandb_project_title (str): The title of the WandB project to use for tracking the experiment.
        
        
    Examples:
        To train a model using a clinical data model with the specified hyperparameters:
        $ python3 training_unimodal_hyper.py clinical /path/to/data /path/to/save/model /path/to/config wandb_project_title
        
    """
    args = sys.argv[1:]
    modality_name = args[0]
    data_path = args[1]
    save_model_path = args[2]
    config_path = args[3]
    wandb_project_title = args[4]
    
    sweep_config = json.load(open(config_path + "config.json"))

    pprint.pprint(sweep_config)
    
    project_name = wandb_project_title + "-" + modality_name
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    task = "phenotyping"
    
    train_inputs = load_dataset(f'{data_path}/{task}/{"train"}_{modality_name}_dataset.pkl')
    val_inputs = load_dataset(f'{data_path}/{task}/{"val"}_{modality_name}_dataset.pkl')
    
    criterion = nn.BCEWithLogitsLoss()
    
    def main_train(config=None):
    # Initialize a new wandb run
        with wandb.init(config=config):
            config = wandb.config

            train_dataloader = DataLoader(train_inputs, batch_size=config.batch_size,shuffle=False) #super important not to shuffle anything in the multi-modal setting

            val_dataloader= DataLoader(val_inputs, batch_size=config.batch_size, shuffle=False)

            dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}
            
            #Here we can also add a folder to store all the models, so this is just an example path
            path = save_model_path + '/models/model_pheno_' + str(config.learning_rate) + "_" + str(config.random_seed) + "_" + str(config.batch_size) + "_" + str(config.epochs) + "_" + modality_name 
             
            train_model(modality_name, dataloaders_dict, criterion, config, path)  
            
    wandb.agent(sweep_id, main_train, count=55) #55 is the number of runs from the combinations of the config

if __name__ == "__main__":
    main()
    