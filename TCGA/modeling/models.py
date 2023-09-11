import sys
sys.path.append('../../..')
import torch                    
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from itertools import combinations
from common_files.models import OvOAttention, MultiHeadAttention
    
class MultimodalFramework(nn.Module):
    """
    A PyTorch module for a multimodal framework for the TCGA dataset.

    Args:
    - modality_dims (List[int]): A list of dimensions for each modality. Only tabular modalities are needed.
    - num_heads (int): The number of heads to use in the multihead attention layers.

    """

    def __init__(self, modality_dims, num_heads):
        super().__init__()
        self.num_mod = 5
        self.num_heads = num_heads
        self.modality_dims = modality_dims 
        
        #input layers
        self.fc1a = nn.Linear(self.modality_dims[0], 256)
        self.fc2a = nn.Linear(self.modality_dims[1], 256)
        self.fc3a = nn.Linear(self.modality_dims[2], 256)
        self.fc4a = nn.Linear(self.modality_dims[3], 256)
        
        ##MLP
        self.fc2b = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        
        ##CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20,
            kernel_size=(5, 5))
        self.bn_1 = nn.BatchNorm2d(20)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
            kernel_size=(5, 5))
        self.bn_2 = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100,
            kernel_size=(5, 5))

        self.fc1 = nn.Linear(in_features= 49950, out_features=128)         
        self.fc2 = nn.Linear(in_features=128, out_features=256)

        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.dropout3 = nn.Dropout(p=0.1)
        
        
        #attention
        self.pairwise_attention  = nn.MultiheadAttention(256, self.num_heads, batch_first = True)
        self.early_attention  = nn.MultiheadAttention(256*self.num_mod, self.num_heads, batch_first = True)
        self.OvO_attention = MultiHeadAttention(256,self.num_heads) 
        
        #out
        self.out_concat = nn.Linear(256*self.num_mod, 5)
        self.out_pairwise = nn.Linear(256*self.num_mod*(self.num_mod-1), 5)
        self.out_OvO = nn.Linear(256*self.num_mod, 5)
        
        
    def bi_directional_att(self, l):

        # All possible pairs in list
        a = list(range(len(l)))
        pairs = list(combinations(a, r=2))
        combos = []
        for pair in pairs:
            #(0,1)
            index_1 = pair[0]
            index_2 = pair[1]
            x = l[index_1]
            y = l[index_2]
            attn_output_LV, attn_output_weights_LV = self.pairwise_attention(x, y, y)
            attn_output_VL, attn_output_weights_VL = self.pairwise_attention(y, x, x)
            combined = torch.cat((attn_output_LV,
                                  attn_output_VL), dim=1)
            combos.append(combined)
        return combos


    def forward(self, inp, model):
        """
        Args:
            - inp: A list of inputs that contains four tensors representing textual features (t1, t2, t3, t4) and one tensor
            representing an image tensor (x).
            - model: A string indicating the type of model to use for the forward pass. It can be "concat", "pairwise", "early", or "OvO".

        """
        t1, t2, t3, t4,  x = inp 
        
        t1 = self.fc1a(t1.to(torch.float32))
        t2 = self.fc2a(t2.to(torch.float32))
        t3 = self.fc3a(t3.to(torch.float32))
        t4 = self.fc4a(t4.to(torch.float32))
        
        outputs = []
        for t in [t1, t2, t3, t4]: 
            t = self.relu(t)
            t = self.dropout2(t)
            t = self.fc2b(t)
            t = self.relu(t)
            t = self.dropout3(t)
            outputs.append(t)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn_1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn_2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
      
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        outputs.append(x)
            
        if model == "concat":
            combined = torch.cat(outputs, dim=1)
            out = self.out_concat(combined)
            
        elif model == "pairwise":
            combined = self.bi_directional_att(outputs)
            comb = torch.cat(combined, dim=1)
            out = self.out_pairwise(comb)
        
        elif model == "early":
            combined = torch.cat(outputs, dim=1)
            comb = self.early_attention(combined, combined, combined)
            out = self.out_concat(comb)
        
        else:
            attns = []
            for main in outputs:
                others = list(set(outputs) - set([main]))
                att = self.OvO_attention(others, main)
                attns.append(att.squeeze(1))
            comb = torch.cat(attns, dim=1)
            out = self.out_OvO(comb)

        return out
    

class UnimodalFramework(nn.Module):
    """
    A PyTorch module for a unimodal framework for the TCGA dataset.

    Args:
    - feature_dim (int): The feature demension of each modality, i.e. if a tensor is of shape (batch_size, feature_dim), feautre_dim is what is being passed in. 
    """

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        ##MLP
        self.fc1a = nn.Linear(self.feature_dim, 256)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2a = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3a = nn.Linear(256, 5)
        self.bn1 = nn.BatchNorm1d(256)
        
        ##CNN
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20,
            kernel_size=(5, 5))
        self.bn_1 = nn.BatchNorm2d(20)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout1 = nn.Dropout(p=0.3)
        
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50,
            kernel_size=(5, 5))
        self.bn_2 = nn.BatchNorm2d(50)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.conv3 = nn.Conv2d(in_channels=50, out_channels=100,
            kernel_size=(5, 5))
        self.bn_3 = nn.BatchNorm2d(100)
        
        
        self.fc1 = nn.Linear(in_features= 49950, out_features=128) 
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.dropout3 = nn.Dropout(p=0.1)
        
        self.fc4 = nn.Linear(in_features=256, out_features=5)
        

    def forward(self, x, modality_name):
        """
        Args:
            - x (torch.Tensor): The input tensor of shape `(batch_size, channels, height, width)` if image, or `(batch_size, feature_dim)` if tabular.
            - modality_name (str): The modality name, either "image" or any other non-image modality name (e.g. "clinical", "epigenomic", "cnv", etc.).

        """
        if modality_name != "image":
            x = x.to(torch.float32)
            x = self.fc1a(x)
            x = self.relu(x)
            x = self.drop1(x)
            
            
            x = self.fc2a(x)
            x = self.relu(x)
            x = self.drop2(x)
            out = self.fc3a(x)

        else:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.bn_1(x)
            x = self.maxpool1(x)
            x = self.dropout1(x)
            
            x = self.conv2(x)
            x = self.relu(x)
            x = self.bn_2(x)
            x = self.maxpool2(x)
            x = self.dropout2(x)
            
            x = flatten(x, 1)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout2(x)
            
            x = self.fc2(x)
            x = self.relu(x)
            x = self.dropout3(x)
            
            x = self.fc3(x)
            x = self.relu(x)
            x = self.dropout3(x)
            
            out = self.fc4(x)
            
        return out
