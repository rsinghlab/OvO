import sys
sys.path.append('../..')
import torch                    
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
from common_files.model_utils import OvOAttention, MultiHeadAttention

class MultimodalFramework(nn.Module):
    def __init__(self, modality_dims, num_heads):
        super().__init__()
        self.num_mod = len(modality_dims)
        self.num_heads = num_heads
        self.modality_dims = modality_dims  # Dictionary of dimensions for each modality
        self.num_labels = 2

        # Encoders for each modality
        self.modality_encoders = nn.ModuleDict({
            "demographics": self.create_encoder(modality_dims["demographics"], 'small'),
            "diagnosis": self.create_encoder(modality_dims["diagnosis"], 'medium'),
            "treatment": self.create_encoder(modality_dims["treatment"], 'large'),
            "medication": self.create_encoder(modality_dims["medication"], 'small'),
            "lab": self.create_encoder(modality_dims["lab"], 'large'),
            "aps": self.create_encoder(modality_dims["aps"], 'small')
        })

        # Attention layers
        self.vaswani_attention = nn.MultiheadAttention(256, num_heads, batch_first=True)
        self.single_attention = nn.MultiheadAttention(self.num_mod * 256, num_heads, batch_first=True)
        self.OvO_attention = MultiHeadAttention(256,self.num_heads) 
        
        # Output layers
        self.out_concat = nn.Linear(256 * self.num_mod, self.num_labels)
        self.out_vaswani = nn.Linear(256 * self.num_mod * (self.num_mod - 1), self.num_labels)
        self.out_single = nn.Linear(256 * self.num_mod, self.num_labels)
        self.out_OvO = nn.Linear(256*self.num_mod, self.num_labels)

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
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
        elif size == "medium":
            return nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.BatchNorm1d(256),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )

        elif size == "large":
            return nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU()
            )
        else:
            raise ValueError(f"Unknown size: {size}")
            
    def forward(self, inp, model):
        demographics_input, diagnosis_input, treatment_input, medication_input, lab_input, aps_input = inp

        # Process each modality using its specific encoder
        demographics_out = self.modality_encoders["demographics"](demographics_input)
        diagnosis_out = self.modality_encoders["diagnosis"](diagnosis_input)
        treatment_out = self.modality_encoders["treatment"](treatment_input)
        medication_out = self.modality_encoders["medication"](medication_input)
        lab_out = self.modality_encoders["lab"](lab_input)
        aps_out = self.modality_encoders["aps"](aps_input)

        # Combine outputs for different model types
        outputs = [demographics_out, diagnosis_out, treatment_out, medication_out, lab_out, aps_out]

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
        self.num_labels = 2

        # Common layers
        self.fc1_small = nn.Linear(self.feature_dim, 128)
        self.fc1_large = nn.Linear(self.feature_dim, 256)
        self.fc2_small = nn.Linear(128, self.num_labels)
        self.fc2_medium = nn.Linear(256, 128)
        self.fc2_large = nn.Linear(256, self.num_labels)
        self.fc3_medium = nn.Linear(128, self.num_labels)

        # Batch normalization and other layers
        self.bn1_small = nn.BatchNorm1d(128)
        self.bn1_large = nn.BatchNorm1d(256)
        self.bn2_medium = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()

    def forward(self, x, modality_name):
        if modality_name in ['demographics', 'medication', 'aps']:
            x = self.fc1_small(x)
            x = self.bn1_small(x)
            x = self.relu(x)
            out = self.fc2_small(x)
        elif modality_name in ['lab', 'treatment']:
            x = self.fc1_large(x)
            x = self.bn1_large(x)
            x = self.relu(x)
            out = self.fc2_large(x)
        elif modality_name == 'diagnosis':
            x = self.fc1_large(x)
            x = self.bn1_large(x)
            x = self.relu(x)
            x = self.fc2_medium(x)
            x = self.bn2_medium(x)
            x = self.relu(x)
            out = self.fc3_medium(x)
        else:
            raise ValueError(f"Unknown modality: {modality_name}")

        return out
