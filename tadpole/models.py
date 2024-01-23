import sys
sys.path.append('../..')
import torch                    
import torch.nn as nn
import torch.nn.functional as F
from torch import flatten
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from itertools import combinations
from common_files.model_utils import OvOAttention, MultiHeadAttention


class MultimodalFramework(nn.Module):
    def __init__(self, modality_dims, num_heads):
        super().__init__()
        self.num_mod = len(modality_dims)
        self.num_heads = num_heads
        self.num_labels = 3
        
        # Encoders for each modality
        self.modality_encoders = nn.ModuleDict({
            "mri": self.create_encoder(modality_dims["mri"], "small"),
            "fdg_pet": self.create_encoder(modality_dims["fdg_pet"], "large"),
            "av45_pet": self.create_encoder(modality_dims["av45_pet"], "medium"),
            "csf": self.create_encoder(modality_dims["csf"], "small"),
            "cognitive_tests": self.create_encoder(modality_dims["cognitive_tests"], "small"),
            "clinical": self.create_encoder(modality_dims["clinical"], "small")
        })

         # Attention layers
        self.vaswani_attention = nn.MultiheadAttention(64, num_heads, batch_first=True)
        self.single_attention = nn.MultiheadAttention(self.num_mod * 64, num_heads, batch_first=True)
        self.OvO_attention = MultiHeadAttention(64,self.num_heads) 
        
        # Output layers
        self.out_concat = nn.Linear(64 * self.num_mod, self.num_labels)
        self.out_vaswani = nn.Linear(64 * self.num_mod * (self.num_mod - 1), self.num_labels)
        self.out_single = nn.Linear(64 * self.num_mod, self.num_labels)
        self.out_OvO = nn.Linear(64*self.num_mod, self.num_labels)

    def bi_directional_att(self, l):
        # All possible pairs in list
        pairs = list(combinations(range(len(l)), 2))
        combos = []
        for pair in pairs:
            index_1, index_2 = pair
            x, y = l[index_1], l[index_2]
            attn_output_LV, attn_output_weights_LV = self.vaswani_attention(x, y, y)
            attn_output_VL, attn_output_weights_VL = self.vaswani_attention(y, x, x)
            combined = torch.cat((attn_output_LV, attn_output_VL), dim=1)
            combos.append(combined)
        return combos


    def create_encoder(self, feature_dim, size):
        if size == "small":
            return nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        elif size == "medium":
            return nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 64)
            )
        elif size == "large":
            return nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )
        else:
            raise ValueError(f"Unknown size: {size}")


    def forward(self, inp, model):

        mri_input, fdg_pet_input, av45_pet_input, csf_input, cognitive_tests_input, clinical_input = inp

        # Process each modality using its specific encoder
        mri_out = self.modality_encoders["mri"](mri_input)
        fdg_pet_out = self.modality_encoders["fdg_pet"](fdg_pet_input)
        av45_pet_out = self.modality_encoders["av45_pet"](av45_pet_input)
        csf_out = self.modality_encoders["csf"](csf_input)
        cognitive_tests_out = self.modality_encoders["cognitive_tests"](cognitive_tests_input)
        clinical_out = self.modality_encoders["clinical"](clinical_input)

        outputs = [mri_out, fdg_pet_out, av45_pet_out, csf_out, cognitive_tests_out, clinical_out]

        if model == "concat":
            combined = torch.cat(outputs, dim=1)
            out = self.out_concat(combined)

        elif model == "cross":
            combined = self.bi_directional_att(outputs)
            comb = torch.cat(combined, dim=1)
            out = self.out_vaswani(comb)

        elif model == "self":
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
        self.num_labels = 3

        # Common layers for all modalities
        self.fc1_small = nn.Linear(self.feature_dim, 128)
        self.fc1_large = nn.Linear(self.feature_dim, 256)
        self.fc2_small = nn.Linear(128, self.num_labels)
        self.fc2_medium = nn.Linear(256, 64)
        self.fc2_large = nn.Linear(256, self.num_labels)
        self.fc3_medium = nn.Linear(64, self.num_labels)

        # Batch normalization and other layers
        self.bn1_small = nn.BatchNorm1d(128)
        self.bn1_large = nn.BatchNorm1d(256)
        self.bn2_medium = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, modality_name):
        if modality_name in ["mri", "cognitive_tests", "clinical", "csf"]:
            x = self.fc1_small(x)
            x = self.bn1_small(x)
            x = self.relu(x)
            out = self.fc2_small(x)
        elif modality_name == "av45_pet":
            x = self.fc1_large(x)
            x = self.bn1_large(x)
            x = self.relu(x)
            x = self.fc2_medium(x)
            x = self.bn2_medium(x)
            x = self.relu(x)
            x = self.dropout(x)
            out = self.fc3_medium(x)
        elif modality_name == "fdg_pet":
            x = self.fc1_large(x)
            x = self.bn1_large(x)
            x = self.relu(x)
            out = self.fc2_large(x)
        else:
            raise ValueError(f"Unknown modality: {modality_name}")

        return out

        

