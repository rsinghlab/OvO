import numpy as np
import json
import sys
sys.path.append('../..')
import logging
import torch                    
import torch.nn as nn
import torchvision
import time
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import wandb
from models import MultimodalFramework
from common_files.model_utils import set_seed, build_optimizer
from common_files.custom_sets import *
from data_utils import *
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

def train_model(model_name, dataloaders, criterion, len_train, len_val, modality_shapes, config, path):
    """
    Trains a multimodal deep learning model using the given data loaders, criterion, and optimizer, and wandb config. Logs metrics to wandb (weights and biases) for hyperparameter tuning.

    Args:
        model_name (str): The name of the multimodal model to be trained.
        dataloaders (Dict[str, list of DataLoaders]): A dictionary containing a list of PyTorch DataLoader objects for the 'train' and 'val' sets.
        criterion (Callable): The loss function to optimize during training.
        len_train (int): length of the train set.
        len_val (int): length of the validation set.
        config: wandb config with hyperparameter values.
        path (str): The file path where the trained model will be saved.

    Returns:
        model (nn.Module): The trained multimodal model.
        history (Dict[str, List[float]]): A dictionary containing training and validation accuracy and loss history.
    """

    set_seed(config.random_seed)
    model = MultimodalFramework(modality_shapes, config.num_heads)
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    num_epochs = config.epochs
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, 0.9)

    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_acc = 0.0
    patience = 10 
    trigger = 0
    acc_dict = {}
    
    for epoch in range(num_epochs):
        #scheduler.step()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                length = len_train
                model.train()  # Set model to training mode
            else:
                length = len_val
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            preds_list = []
            labels_list =[]

            for i, data in enumerate(dataloaders[phase]):
                optimizer.zero_grad()  # zero the parameter gradients

                # Extract modalities data
                text, img, ts, demo = data
                text_input, masks, text_labels = text
                img_input, img_labels = img
                ts_input, ts_labels = ts
                demo_input, demo_labels = demo

                # Move data to device
                text_input, masks, img_input, ts_input, demo_input, labels  = text_input.to(device), masks.to(device), img_input.to(device), ts_input.to(device), demo_input.to(device), text_labels.to(device)
                
                inp_len = labels.size(0)
                # forward
                with torch.set_grad_enabled(phase == 'train'):
                    # Pass all modality inputs to model
                    outputs = model([(text_input, masks), img_input, ts_input, demo_input], model_name)

                    # Compute loss
                    loss = criterion(outputs, labels.float())
                    #_, preds = torch.max(outputs, 1)
                    preds = outputs.sigmoid() > 0.5

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inp_len
                running_corrects += torch.sum(preds == labels.squeeze())
                labels_list.extend(labels.cpu())
                preds_list.extend(preds.cpu())

            epoch_loss = running_loss / length
            epoch_acc = running_corrects.double() / length
            epoch_f1 = f1_score(labels_list, preds_list, average='macro')
            epoch_auroc = roc_auc_score(labels_list, preds_list, average='macro')
            epoch_auprc = average_precision_score(labels_list, preds_list, average='macro')

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc, "val_f1": epoch_f1, "val_auroc": epoch_auroc, "val_auprc": epoch_auprc})
                acc_dict[epoch] = float(epoch_f1)
                torch.save(model.state_dict(), path + "_current.pth")
                if epoch_f1 > best_acc:
                    best_acc = epoch_f1
                    torch.save(model.state_dict(), path+"_best.pth")
    
                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger +=1
                    if trigger >= patience:
                        return model, {"train_acc":train_acc_history, "val_acc":val_acc_history,"train_loss":train_loss_history, "val_loss":val_loss_history}
                else:
                    trigger = 0
 
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc,"train_f1": epoch_f1, "train_auroc":epoch_auroc, "train_auprc":epoch_auprc, "epoch": epoch})


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model, {"train_acc":train_acc_history, "val_acc":val_acc_history,"train_loss":train_loss_history, "val_loss":val_loss_history}

         
        
def main():
    
    """
    Trains a mulitmodal PyTorch model with Weights and Biases (wandb) and hyperparameter tuning.

    Arguments:
        model_name (str): The name of the model architecture to use (e.g, concatenation, OvO, cross, self).
        data_path (str): The path to the directory containing the training and validation data.
        save_model_path (str): The path to the directory where the trained model will be saved.
        config_path (str): The path to the directory containing the configuration file for the hyperparameter search.
        wandb_project_title (str): The title of the WandB project to use for tracking the experiment.
        
        
    Examples:
        To train a multimodal model using simple concatenation with the specified hyperparameters:
        $ python3 training_multimodal_hyper.py concat /path/to/data /path/to/save/model /path/to/config wandb_project_title
        
        To train a multimodal model with OvO attention architecture with the specified hyperparameters:
        $ python3 training_multimodal_hyper.py OvO /path/to/data /path/to/save/model /path/to/config wandb_project_title
        
    """
    args = sys.argv[1:]
    model_name = args[0]
    data_path = args[1]
    save_model_path = args[2]
    config_path = args[3]
    wandb_project_title = args[4]
    
    sweep_config = json.load(open(config_path + "config.json"))
    
    project_name = wandb_project_title + "-" + model_name

    sweep_id = wandb.sweep(sweep_config, project=project_name)
    modalities = ["text", "img", "ts", "demo"]
    task = "phenotyping"  
    criterion = nn.BCEWithLogitsLoss()
    
    def main_train(config=None):
    # Initialize a new wandb run
        with wandb.init(config=config):
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = wandb.config
            
            train_input_list = []
            val_input_list = []
            modality_shapes = {}

            for modality_name in modalities:

                train_inputs = load_dataset(f'{data_path}/{task}/{"train"}_{modality_name}_dataset.pkl')
                val_inputs = load_dataset(f'{data_path}/{task}/{"val"}_{modality_name}_dataset.pkl')

                if modality_name == "ts":
                    numcols = next(iter(train_inputs))[0].shape[1]
                    print(numcols)
                else:
                    numcols = next(iter(train_inputs))[0].shape[0]
                    print(numcols)

                modality_shapes[modality_name] = numcols
                train_input_list.append(DataLoader(train_inputs, batch_size=config.batch_size,shuffle=False))
                val_input_list.append(DataLoader(val_inputs, batch_size=config.batch_size, shuffle=False))

                #this is a bit dumb since the value will be the same for each modalities, but that's my temporary solution
                len_val = len(val_inputs)
                len_train = len(train_inputs)

            train_loader = MyLoader(train_input_list)
            val_loader = MyLoader(val_input_list)
            
            dataloaders_dict = {'train':train_loader, 'val':val_loader}
            
            path = save_model_path + '/models/model_' + str(config.learning_rate) + "_" + str(config.batch_size) + "_" + str(config.random_seed) + "_" + str(config.epochs)+ "_" + str(config.num_heads) + "_" + model_name 
             
            train_model(model_name, dataloaders_dict, criterion, len_train, len_val, modality_shapes, config, path)  
            
    wandb.agent(sweep_id, main_train, count=200)

if __name__ == "__main__":
    main()
    
