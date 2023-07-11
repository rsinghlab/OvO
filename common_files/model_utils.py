import numpy as np
import random
import torch                    
import torch.optim as optim
import os
from PIL import ImageFile
from transformers import BertTokenizer
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def tokenize_mask(sentences, labels):
    """
    Tokenizes a list of sentences using the BERT tokenizer and creates attention masks.
    
    Args:
    - sentences (list of str): A list of sentences to tokenize.
    - labels (list or array-like): Labels for the corresponding sentences.
    
    Returns:
    - input_ids (torch.Tensor): A tensor of input IDs for the tokenized sentences.
    - attention_masks (torch.Tensor): A tensor of attention masks for the tokenized sentences.
    - labels (torch.Tensor): A tensor of labels for the corresponding sentences.
    """

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    input_ids = []
    attention_masks = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent,                      
                            add_special_tokens = True, 
                            max_length = 100,          
                            pad_to_max_length = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',   
                       )
   
        input_ids.append(encoded_dict['input_ids'])

        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels
    
def build_optimizer(network, optimizer, learning_rate):
    """
    Builds and returns an optimizer for the given model.
    
    Args:
    - network (torch.nn.Module): The model to optimize.
    - optimizer (str): The type of optimizer to use. Must be one of "sgd", "adam", or "adamW".
    - learning_rate (float): The learning rate for the optimizer.
    
    Returns:
    - optimizer (torch.optim.Optimizer): The optimizer for the given network.
    """

    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    elif optimizer == "adamW":
        optimizer = optim.AdamW(network.parameters(),
                               lr=learning_rate, eps = 1e-8)
    return optimizer



