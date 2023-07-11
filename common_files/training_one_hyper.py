import sys
import logging
import json
import torch                    
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import time
import pprint
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import wandb

from models import MultimodalFramework
from model_utils import set_seed, build_optimizer
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

def train_model(model_name, dataloaders, criterion, config, path):
    """
    Trains a unimodal deep learning model using the given data loaders, criterion, and optimizer, and wandb config. Logs metrics to wandb (weights and biases) for hyperparameter tuning.

    Args:
        model_name (str): The name of the unimodal model to be trained (e.g., ResNet, Bert).
        dataloaders (Dict[str, DataLoader]): A dictionary containing PyTorch DataLoader objects for the 'train' and 'val' sets.
        criterion (Callable): The loss function to optimize during training.
        config: wandb config with hyperparameter values.
        path (str): The file path where the trained model will be saved.

    Returns:
        model (nn.Module): The trained unimodal model.
        history (Dict[str, List[float]]): A dictionary containing training and validation accuracy and loss history.
    """
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(config.random_seed)
    model = MultimodalFramework(num_heads = 1, num_mod = 1) #this only calls the unimodal framework inside "MultimodalFramework"
    
    model.to(device)
    
    num_epochs = config.epochs
    optimizer = build_optimizer(model, config.optimizer,config.learning_rate)

    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

 
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()  

            running_loss = 0.0
            running_corrects = 0


            for data in dataloaders[phase]:
                if model_name == "bert":
                    inputs, masks, labels = data
                    inputs, masks, labels = inputs.to(device), masks.to(device), labels.to(device)
                elif model_name == "mlp":
                    inputs, labels = data
                    inputs, labels =inputs.float(), labels.long()
                    inputs, labels = inputs.to(device), labels.to(device)
                else:
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    
                    if model_name == "bert":
                        outputs = model([inputs, masks], model_name)
      
                    else:
                        outputs = model(inputs, model_name)
                    
                    
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()


                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val':
                wandb.log({"val_loss": epoch_loss, "val_acc": epoch_acc})
                val_acc_history.append(epoch_acc.detach().cpu())
                val_loss_history.append(epoch_loss) 
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), path)
                    
            if phase == 'train':
                wandb.log({"train_loss": epoch_loss, "train_acc": epoch_acc, "epoch": epoch})
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
        model_name (str): The name of the model architecture to use.
        data_path (str): The path to the directory containing the training and validation data.
        save_model_path (str): The path to the directory where the trained model will be saved.
        config_path (str): The path to the directory containing the configuration file for the hyperparameter search.
        wandb_project_title (str): The title of the WandB project to use for tracking the experiment.
        
        
    Examples:
        To train a model using a BERT model with the specified hyperparameters:
        $ python3 training_one_hyper.py bert /path/to/data /path/to/save/model /path/to/config wandb_project_title
        
    """
    args = sys.argv[1:]
    model_name = args[0]
    data_path = args[1]
    save_model_path = args[2]
    config_path = args[3]
    wandb_project_title = args[4]
    
    sweep_config = json.load(open(config_path + "config.json"))

    pprint.pprint(sweep_config)
    
    project_name = wandb_project_title + "-" + model_name
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    if model_name == "bert":
        train_inputs = torch.load(data_path + 'train_txt_inputs.pt')
        val_inputs = torch.load(data_path + 'val_txt_inputs.pt')
        
    elif model_name == "mlp":
        train_inputs = torch.load(data_path + 'train_tab_inputs.pt')
        val_inputs = torch.load(data_path + 'val_tab_inputs.pt')
        
    else:
        train_inputs = torch.load(data_path + 'train_img_inputs.pt')
        val_inputs = torch.load(data_path + 'val_img_inputs.pt')

    
    
    def main_train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            train_dataloader = DataLoader(train_inputs, batch_size=config.batch_size,shuffle=False)

            val_dataloader= DataLoader(val_inputs, batch_size=config.batch_size, shuffle=False)

            dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}
            
            path = save_model_path + 'model_' + str(config.learning_rate) + "_" + str(config.random_seed) + "_" + str(config.optimizer) + "_" + str(config.epochs) + "_" + model_name +".pth"

            criterion = nn.CrossEntropyLoss()
            train_model(model_name, dataloaders_dict, criterion, config, path)  
            
    wandb.agent(sweep_id, main_train, count=100)

if __name__ == "__main__":
    main()
    
