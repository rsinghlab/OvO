import sys
sys.path.append('../..')
import logging
import torch 
import json                   
import torch.nn as nn
import torchvision
import time
import pprint
from torch.utils.data import DataLoader
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)
import wandb
from common_files.models import MultimodalFramework
from common_files.model_utils import set_seed, build_optimizer
from common_files.custom_sets import MyLoader
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

torch.cuda.empty_cache()

def train_model(model_name, dataloaders, criterion, len_train, len_val, config, path):
    """
    Trains a multimodal deep learning model using the given data loaders, criterion, and optimizer, and wandb config. Logs metrics to wandb (weights and biases) for hyperparameter tuning.

    Args:
        model_name (str): The name of the multimodal model to be trained (e.g., resnet_bert_mlp, resnet_bert__mlp_OvO, resnet_bert_mlp_pairwise ).
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
    model = MultimodalFramework(num_heads = config.num_heads, num_mod = 3)
    
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)
    
    num_epochs = config.epochs
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate)

    since = time.time()

    val_acc_history = []
    val_loss_history = []
    train_acc_history = []
    train_loss_history = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
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
                    img, text, tab = data
                    
                    text_inp, masks, text_labels = text
                    img_inp, labels = img

                    text_inp, masks, text_labels = text_inp.to(device), masks.to(device), text_labels.to(device)
                    img_inp, labels = img_inp.to(device), labels.to(device)

                    tab_inp, tab_labels = tab
                    tab_inp, tab_labels = tab_inp.float(), tab_labels.long()

                    tab_inp, tab_labels = tab_inp.to(device), tab_labels.to(device)

                    inp_len = text_inp.size(0)
                    outputs = model([tab_inp, img_inp, text_inp, masks], model_name)

                    loss = criterion(outputs, text_labels)
                    print("here")
                    _, preds = torch.max(outputs, 1)

                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

             
                running_loss += loss.item() * text_inp.size(0)
                running_corrects += torch.sum(preds == text_labels.data)

            epoch_loss = running_loss / length
            epoch_acc = running_corrects.double() / length

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
    Trains a mulitmodal PyTorch model with Weights and Biases (wandb) and hyperparameter tuning.

    Arguments:
        model_name (str): The name of the model architecture to use.
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
    data_path = args[1]
    save_model_path = args[2]
    config_path = args[3]
    wandb_project_title = args[4]

    sweep_config = json.load(open(config_path + "config.json"))

    pprint.pprint(sweep_config)
    
    project_name = wandb_project_title + "-" + model_name
    
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    
    train_inputs_txt = torch.load(data_path + 'train_txt_inputs.pt')
    val_inputs_txt = torch.load(data_path + 'val_txt_inputs.pt')

    train_inputs_tab = torch.load(data_path + 'train_tab_inputs.pt')
    val_inputs_tab = torch.load(data_path + 'val_tab_inputs.pt')

    train_inputs_img = torch.load(data_path + 'train_img_inputs.pt')
    val_inputs_img = torch.load(data_path + 'val_img_inputs.pt')

    
    def main_train(config=None):
        with wandb.init(config=config):
            config = wandb.config

            train_dataloader_text = DataLoader(train_inputs_txt, batch_size=config.batch_size,shuffle=False)
            val_dataloader_text = DataLoader(val_inputs_txt, batch_size=config.batch_size, shuffle=False)

            train_dataloader_tab = DataLoader(train_inputs_tab, batch_size=config.batch_size,shuffle=False)
            val_dataloader_tab = DataLoader(val_inputs_tab, batch_size=config.batch_size, shuffle=False)

            train_dataloader_img = DataLoader(train_inputs_img, batch_size=config.batch_size,shuffle=False)
            val_dataloader_img = DataLoader(val_inputs_img, batch_size=config.batch_size, shuffle=False)
            
            len_val = len(val_inputs_txt)
            len_train = len(train_inputs_txt)
            
            train_loader = MyLoader([train_dataloader_img, train_dataloader_text, train_dataloader_tab])
            val_loader = MyLoader([val_dataloader_img, val_dataloader_text, val_dataloader_tab])
            dataloaders_dict = {'train':train_loader, 'val':val_loader}

            path = save_model_path + 'model_' + str(config.learning_rate) + "_" + str(config.random_seed)  + "_" + str(config.optimizer) + "_" + str(config.epochs) + "_" + model_name +".pth"
            criterion = nn.CrossEntropyLoss()
            train_model(model_name, dataloaders_dict, criterion, len_train, len_val, config, path)  
            
    wandb.agent(sweep_id, main_train, count=200)

if __name__ == "__main__":
    main()
    
