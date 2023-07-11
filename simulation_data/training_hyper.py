import sys
sys.path.append('../..')
import pandas as pd
import torch
import numpy as np
import time
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from common_files.model_utils import MyLoader, set_seed
from models import OvO, pairwise, concat
import wandb


def train_model(model_name, dataloaders, criterion, len_train, len_val,num_modalities, path, config):
    """
    Trains a multimodal deep learning model using the given data loaders, criterion, and optimizer, and wandb config. Logs metrics to wandb (weights and biases) for hyperparameter tuning.

    Args:
        model_name (str): The name of the multimodal model to be trained (e.g., concat, OvO, pairwise ).
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
    
    if model_name == "concat":
        model = concat(num_modalities)
    elif model_name == "pairwise":
        model = pairwise(num_modalities, config.num_heads)
    else:
        model = OvO(num_modalities, config.num_heads)
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    num_epochs = config.epochs
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)

    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience = 10 
    trigger = 0
    acc_dict = {}
    best_epoch = 0
    for epoch in range(num_epochs):
        
        for phase in ['train', 'val']:
            if phase == 'train':
                length = len_train
                model.train()  
            else:
                length = len_val
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            for i, data in enumerate(dataloaders[phase]):
            
                optimizer.zero_grad()

                
                with torch.set_grad_enabled(phase == 'train'):
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

                    loss = criterion(outputs, labels.squeeze().long())
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

             
                running_loss += loss.item() * inp_len
                running_corrects += torch.sum(preds == labels.squeeze())

            epoch_loss = running_loss / length
            epoch_acc = running_corrects.double() / length


            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc})
                acc_dict[epoch] = float(epoch_acc.detach().cpu())
                val_acc_history.append(epoch_acc.detach().cpu())
                val_loss_history.append(epoch_loss) 
                torch.save(model.state_dict(), path+"_current.pth")
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_epoch = epoch
                    torch.save(model.state_dict(), path+"_best.pth")
                #"""    
                if (epoch > 10) and (acc_dict[epoch] <= acc_dict[epoch - 10]):
                    trigger +=1
                    if trigger >= patience:
                        print('Best val Acc: {:4f}'.format(best_acc))
                        model.load_state_dict(best_model_wts)
                        return model, {"train_acc":train_acc_history, "val_acc":val_acc_history,"train_loss":train_loss_history, "val_loss":val_loss_history}
                else:
                    trigger = 0
                #"""   
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc, "epoch": epoch})
                train_acc_history.append(epoch_acc.detach().cpu())
                train_loss_history.append(epoch_loss)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


    model.load_state_dict(best_model_wts)
    return model, {"train_acc":train_acc_history, "val_acc":val_acc_history,"train_loss":train_loss_history, "val_loss":val_loss_history}, best_epoch


def main():
    """
    Trains a mulitmodal PyTorch model with Weights and Biases (wandb) and hyperparameter tuning.

    Arguments:
        model_name (str): The name of the model architecture to use.
        num_modalities (int): Number of simulated modalities to tune. In the paper we used 2, 5, 10, 15, 20, but any number >= 2 works
        data_path (str): The path to the directory containing the training and validation data.
        save_model_path (str): The path to the directory where the trained model will be saved.
        config_path (str): The path to the directory containing the configuration file for the hyperparameter search.
        wandb_project_title (str): The title of the WandB project to use for tracking the experiment.
        
        
    Examples:
        To train a model using a BERT, ResNet, and MLP concatenated with the specified hyperparameters:
        $ python3 training_three_hyper.py bert_resnet_mlp /path/to/data /path/to/save/model /path/to/config wandb_project_title
        
        To train a model using a BERT, ResNet, and MLP with OvO attention architecture with the specified hyperparameters:
        $ python3 training_three_hyper.py bert_resnet_mlp_OvO /path/to/data /path/to/save/model /path/to/config wandb_project_title
        
    """
    args = sys.argv[1:]
    model_name = args[0]
    num_modalities = int(args[1])  
    data_path = args[2]
    save_model_path = args[3]
    config_path = args[4]
    wandb_project_title = args[5]

    sweep_config = json.load(open(config_path + "config.json"))
    project_name = wandb_project_title + "-" + model_name
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    def main_train(config=None):
    # Initialize a new wandb run
        with wandb.init(config=config):

            config = wandb.config
            criterion = nn.CrossEntropyLoss()

            y_train = pd.read_pickle(data_path + "y_train.pkl")
            y_val = pd.read_pickle(data_path + "y_val.pkl")
            
            train_input_list = []
            test_input_list = []
            val_input_list = []
            for j in range(num_modalities):
                train_inputs = torch.load(data_path + "train_modality_" + str(j) +  "_inputs.pt")
                val_inputs = torch.load(data_path + "val_modality_" + str(j) +  "_inputs.pt")
                test_inputs = torch.load(data_path + "test_modality_" + str(j) +  "_inputs.pt")
                train_input_list.append(DataLoader(train_inputs, batch_size=config.batch_size,shuffle=False))
                val_input_list.append(DataLoader(val_inputs, batch_size=config.batch_size, shuffle=False))
                test_input_list.append(DataLoader(test_inputs, batch_size=config.batch_size, shuffle=False))
            train_loader = MyLoader(train_input_list)
            val_loader = MyLoader(val_input_list)

            dataloaders_dict = {'train':train_loader, 'val':val_loader}
            len_val = len(y_val)
            len_train = len(y_train)
            
            model_path = save_model_path + '/model_' + str(config.learning_rate) + "_" + str(config.batch_size) + "_" + str(config.random_seed) + "_" + str(config.epochs)+ "_" + str(config.num_heads) + "_" + model_name + "_" + str(num_modalities)
            train_model(model_name, dataloaders_dict, criterion, len_train, len_val, num_modalities, model_path, config)  
            
    wandb.agent(sweep_id, main_train, count=200)


if __name__ == "__main__":
    main()
    
