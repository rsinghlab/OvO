import sys
sys.path.append('../..')
import torch                    
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from itertools import combinations
from transformers import AutoModel
from common_files.model_utils import OvOAttention, MultiHeadAttention

class MultimodalFramework(nn.Module):

    def __init__(self, modality_dims, num_heads):
        super().__init__()
        self.num_mod = len(modality_dims)
        self.num_heads = num_heads
        self.modality_dims = modality_dims #list of dimensions for each modality (only tabular)
        self.num_labels = 25
        
        # ClinicalBERT for text
        self.text_model_name = "medicalai/ClinicalBERT"
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.text_classifier = nn.Linear(self.text_model.config.hidden_size, 256)

        # CNN for images
        self.image_encoder = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(p=0.3),

            # Convolutional Layer 2
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(50),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(p=0.2),

            # Flatten the output
            nn.Flatten(),

            # Fully Connected Layer 1
            nn.Linear(in_features=140450, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            # Fully Connected Layer 2
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            # Fully Connected Layer 3
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        # LSTM for time-series
        self.lstm = nn.LSTM(input_size=self.modality_dims['ts'], hidden_size=256, num_layers=1, batch_first=True)
        self.fc_lstm = nn.Linear(256, 256)

        # MLP for tabular data
        self.mlp_encoder = nn.Sequential(
            nn.Linear(modality_dims["demo"], 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            #nn.Linear(256, num_labels)
        )
        
        
        #attention
        self.vaswani_attention  = nn.MultiheadAttention(256, self.num_heads, batch_first = True)
        self.single_attention  = nn.MultiheadAttention(self.num_mod * 256, self.num_heads, batch_first = True)
        self.OvO_attention = MultiHeadAttention(256,self.num_heads) 
        
        #out
        self.out_concat = nn.Linear(256*self.num_mod, self.num_labels)
        self.out_vaswani = nn.Linear(256*self.num_mod*(self.num_mod-1), self.num_labels)
        self.out_single = nn.Linear(256*self.num_mod, self.num_labels)
        self.out_OvO = nn.Linear(256*self.num_mod, self.num_labels)
        
        
    def bi_directional_att(self, l):

        # All possible pairs in list
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
            attn_output_LV, attn_output_weights_LV = self.vaswani_attention(x, y, y)
            attn_output_VL, attn_output_weights_VL = self.vaswani_attention(y, x, x)
            combined = torch.cat((attn_output_LV,
                                  attn_output_VL), dim=1)
            combos.append(combined)
        return combos


    def forward(self, inp, model):
        #x is a list of inputs
        text_input, image_input, ts_input, demo_input = inp  # Assuming this is how inputs are structured

        # Text modality - ClinicalBERT
        text, masks = text_input
        text_out = self.text_model(text, attention_mask=masks).last_hidden_state[:,0,:]
        text_out = self.text_classifier(text_out)

        # Image modality - CNN
        image_out = self.image_encoder(image_input)

        # Time-series modality - LSTM
        ts_out, _ = self.lstm(ts_input)
        ts_out = self.fc_lstm(ts_out[:, -1, :])

        # Tabular modality - MLP
        demo_out = self.mlp_encoder(demo_input.float())
        
        outputs = [text_out, image_out, ts_out, demo_out]
            
        if model == "concat":
            combined = torch.cat(outputs, dim=1)
            out = self.out_concat(combined)
            
        elif model == "cross":
            combined = self.bi_directional_att(outputs)
            comb = torch.cat(combined, dim=1)
            out = self.out_vaswani(comb)
            
        elif model == "self":
            #combined = self.bi_directional_att(outputs)
            combined = torch.cat(outputs, dim=1)
            attn_output, attn_output_weights = self.single_attention(combined, combined, combined)
            out = self.out_single(attn_output)
        
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

    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.text_model_name = "medicalai/ClinicalBERT"
        self.num_labels = 2
        
        ##MLP
        self.fc1a = nn.Linear(self.feature_dim, 256)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.fc2a = nn.Linear(256, 256)
        self.drop2 = nn.Dropout(p=0.2)
        self.fc3a = nn.Linear(256, self.num_labels)
        self.bn1 = nn.BatchNorm1d(256)
        
        ##ClinicalBERT
        self.text_model = AutoModel.from_pretrained(self.text_model_name)
        self.text_classifier = nn.Linear(self.text_model.config.hidden_size, self.num_labels)

        
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
        
        
        self.fc1 = nn.Linear(in_features= 140450, out_features=128) 
        self.bn_4 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(in_features=128, out_features=256)
        self.bn_5 = nn.BatchNorm1d(256)

        
        self.fc3 = nn.Linear(in_features=256, out_features=256)
        self.bn_6 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(p=0.1)
        
        self.fc4 = nn.Linear(in_features=256, out_features=self.num_labels)
        
        # for time-series data
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=256, num_layers=2, batch_first=True)
        self.fc_lstm = nn.Linear(256, self.num_labels)
 


    def forward(self, x, modality_name):
        if modality_name == "text":
            text, masks = x
            text_features = self.text_model(text, attention_mask=masks).last_hidden_state[:,0,:]  
            out = self.text_classifier(text_features)

        elif modality_name == "img":
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
            
        elif modality_name == "ts":
            # LSTM for time-series data
            x, _ = self.lstm(x)  # Assuming x is of shape (batch_size, seq_len, features)
            x = x[:, -1, :]  # Get the last time step
            out = self.fc_lstm(x)
            
        else:
            x = x.to(torch.float32)
            x = self.fc1a(x)
            x = self.relu(x)
            x = self.drop1(x)
            
            
            x = self.fc2a(x)
            x = self.relu(x)
            x = self.drop2(x)
            out = self.fc3a(x)
            
        return out
