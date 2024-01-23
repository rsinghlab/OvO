import numpy as np
import json
import logging
import torch                    
import torch.nn as nn
import torchvision
import time
import pprint
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import wandb
from models import UnimodalFramework
import sys
sys.path.append('../..')
from common_files.model_utils import set_seed, build_optimizer
from common_files.custom_sets import *
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
        modality_name (str): The name of the unimodal modality to be trained (e.g., demographics, medication).
        dataloaders (Dict[str, DataLoader]): A dictionary containing PyTorch DataLoader objects for the 'train' and 'val' sets.
        criterion (Callable): The loss function to optimize during training.
        config: wandb config with hyperparameter values.
        path (str): The file path where the trained model will be saved.

    Returns:
        model (nn.Module): The trained unimodal model.
        history (Dict[str, List[float]]): A dictionary containing training and validation accuracy and loss history.
    """
    set_seed(config.random_seed)
    
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
            all_labels = []
            all_predictions = []

            for data in dataloaders[phase]:
                torch.cuda.empty_cache()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    model.to(device)

                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs, modality_name)
                    loss = criterion(outputs, labels.long().squeeze())

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # Convert outputs to probabilities for MAUC
                    probabilities = F.softmax(outputs, dim=1).detach().cpu().numpy()
                    predictions = np.argmax(probabilities, axis=1)
                    labels_np = labels.detach().cpu().numpy()

                    all_labels.append(labels_np)
                    all_predictions.append(predictions)
                    

            # Concatenate all labels and predictions
            all_labels = np.concatenate(all_labels)
            all_predictions = np.concatenate(all_predictions)

            # Calculate metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_f1 = f1_score(all_labels, all_predictions, average='macro')


            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc,  "val_f1": epoch_f1}) 
               
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
                #"""
                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger +=1
                    if trigger >= patience:
                        return model, {"train_acc":train_acc_history, "val_acc":val_acc_history,"train_loss":train_loss_history, "val_loss":val_loss_history}
                else:
                    trigger = 0
                 
                 #"""
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc,"train_f1": epoch_f1, "epoch": epoch}) #this logs to weights and biases
                
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
        To train a model using a demographics model with the specified hyperparameters:
        $ python3 training_unimodal_hyper.py demographics /path/to/data /path/to/save/model /path/to/config wandb_project_title
        
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
    
    train_inputs = torch.load((f'{data_path}/{modality_name}_train.pt'))
    val_inputs = torch.load((f'{data_path}/{modality_name}_val.pt'))

    criterion = nn.CrossEntropyLoss()
    
    def main_train(config=None):
    # Initialize a new wandb run
        with wandb.init(config=config):
            config = wandb.config

            train_dataloader = DataLoader(train_inputs, batch_size=config.batch_size,shuffle=False) #super important not to shuffle anything in the multi-modal setting

            val_dataloader= DataLoader(val_inputs, batch_size=config.batch_size, shuffle=False)

            dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}
            
            #Here we can also add a folder to store all the models, so this is just an example path
            path = save_model_path + '/models/model_mortality_' + str(config.learning_rate) + "_" + str(config.random_seed) + "_" + str(config.batch_size) + "_" + str(config.epochs) + "_" + modality_name 
             
            train_model(modality_name, dataloaders_dict, criterion, config, path)  
            
    wandb.agent(sweep_id, main_train, count=55) #55 is the number of runs from the combinations of the config

if __name__ == "__main__":
    main()
    