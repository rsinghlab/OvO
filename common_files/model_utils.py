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

class OvOAttention(nn.Module):
    """
    Module that implements One-vs-Others attention mechanism as proposed by the paper.
    """
    def __init__(self):
        super(OvOAttention, self).__init__()
    
    def forward(self, others, main, W):
        """
        Compute context vector and attention weights using One-vs-Others attention.

        Args:
            others (List[torch.Tensor]): List of tensors of shape (batch_size, num_heads, seq_len, embed_dim) representing
                                          the other modality inputs.
            main (torch.Tensor): A tensor of shape (batch_size, num_heads, seq_len, embed_dim) representing the main modality input.
            W (torch.nn.Parameter): A learnable parameter tensor of shape (d_head, d_head) representing the weight matrix.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, embed_dim) representing the context vector.
            torch.Tensor: A tensor of shape (batch_size, num_heads, seq_len) representing the attention weights.
        
        """
        mean = sum(others) / len(others)
        score = mean.squeeze(2) @ W @ main.squeeze(2).transpose(1, 2) 
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, main.squeeze(2))
        return context, attn
    
class MultiHeadAttention(nn.Module):
    """
    Module that implements Multi-Head attention mechanism. This was adapted and modified from https://github.com/sooftware/attentions. 

    Args:
        d_model (int): Dimensionality of the input embedding.
        num_heads (int): Number of attention heads.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, seq_len, embed_dim) representing the context vector.
    """
    
    def __init__(self, d_model: int = 512, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        self.d_head = int(d_model / num_heads)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ovo_attn = OvOAttention()
        self.query_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.key_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.value_proj = nn.Linear(d_model, self.d_head * num_heads)
        self.W = torch.nn.Parameter(torch.FloatTensor(self.d_head, self.d_head).uniform_(-0.1, 0.1))

    def forward(self, other, main):
        """
        Compute context vector using Multi-Head attention.

        Args:
            others (List[torch.Tensor]): List of tensors of shape (batch_size, num_heads, seq_len, embed_dim) representing
                                          the other modality inputs.
            main (torch.Tensor): A tensor of shape (batch_size, num_heads, seq_len, embed_dim) representing the main modality input.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, seq_len, embed_dim) representing the context vector.

        """
        batch_size = main.size(0)
        main = main.unsqueeze(1)
        bsz, tgt_len, embed_dim = main.shape
        src_len, _, _ = main.shape
        
        main = main.contiguous().view(tgt_len, bsz * self.num_heads, self.d_head).transpose(0, 1)
        main = main.view(bsz, self.num_heads, tgt_len, self.d_head)
        others = []
        for mod in other:
            mod = mod.unsqueeze(1)
            mod = mod.contiguous().view(tgt_len, bsz * self.num_heads, self.d_head).transpose(0, 1)
            mod = mod.view(bsz, self.num_heads, tgt_len, self.d_head)
            others.append(mod)  
        context, attn = self.ovo_attn(others, main, self.W)
        context = context.contiguous().view(bsz * tgt_len, embed_dim)
        context = context.view(bsz, tgt_len,  context.size(1))
        
        return context


