
import sys
sys.path.append('../..')
import torch                    
import torch.nn as nn
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from itertools import combinations

from common_files.models import OvOAttention, MultiHeadAttention


class concat(nn.Module):

    def __init__(self,num_mod):
        super().__init__()
        self.num_mod = num_mod
        
        
        #input layers
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 256)
        
        
        ##MLP
        #self.fc2b = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        
        #out
        self.out_concat = nn.Linear(256*self.num_mod, 2)
        
    def forward(self, x):
        #x is a list of inputs
        outputs = []
        for i in range(self.num_mod):
            t = x[i]
            t = self.fc1(t.to(torch.float32))
            t = self.fc2(t)
            t = self.relu(t)
            outputs.append(t)

        combined = torch.cat(outputs, dim=1)
        out = self.out_concat(combined)
        
        return out

class pairwise(nn.Module):

    def __init__(self, num_mod, num_heads):
        super().__init__()
        self.num_mod = num_mod
        self.num_heads = num_heads
        
        #input layers
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 256)
        
        ##MLP
        #self.fc2b = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        
        #attention
        self.pairwise_attention  = nn.MultiheadAttention(256, self.num_heads, batch_first = True)
        
        #out
        self.out_pairwise = nn.Linear(256*self.num_mod*(self.num_mod-1), 2)
        
        
    def bi_directional_att(self, l):

        # All possible pairs in List
        # Using combinations()
        a = list(range(len(l)))
        pairs = list(combinations(a, r=2))
        #pairs = torch.combinations(torch.tensor(l), 2)
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


    def forward(self, x):
        #x is a list of inputs
        outputs = []
        for i in range(self.num_mod):
            t = x[i]
            t = self.fc1(t.to(torch.float32))
            t = self.fc2(t)
            t = self.relu(t)
            outputs.append(t)

        combined = self.bi_directional_att(outputs)
        comb = torch.cat(combined, dim=1)
        out = self.out_pairwise(comb)
        return out
    
class OvO(nn.Module):

    def __init__(self, num_mod, num_heads):
        super().__init__()
        self.num_mod = num_mod
        self.num_heads = num_heads
        
        #input layers
        self.fc1 = nn.Linear(20, 256)
        self.fc2 = nn.Linear(256, 256)
        
        ##MLP
        self.fc2b = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        
        #attention
        self.OvO_multihead_attention = MultiHeadAttention(256,self.num_heads)
        
        #out
        self.out_OvO = nn.Linear(256*self.num_mod, 2)
     

    def forward(self, x):
        #x is a list of inputs
        outputs = []
        for i in range(self.num_mod):
            t = x[i]
            t = self.fc1(t.to(torch.float32))
            t = self.fc2(t)
            t = self.relu(t)
            outputs.append(t)


        attns = []
        for main in outputs:
            others = list(set(outputs) - set([main]))
            att = self.OvO_multihead_attention(others, main)
            attns.append(att.squeeze(1))
        comb = torch.cat(attns, dim=1)
        out = self.out_OvO(comb)
        return out
 