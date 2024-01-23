import torch                    
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import OrderedDict
from transformers import BertModel

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from model_utils import OvOAttention, MultiHeadAttention
    
class MultimodalFramework(nn.Module):
    """
    A PyTorch module for a multimodal framework that uses an multilayer perceptron (MLP), ResNet18, and BERT, and their combinations for classification.

    Args:
        num_heads (int): The number of attention heads to use in the pairwsie and OvO attention schemes.
        num_mod (int): The number of modalities (1, 2, or 3).

    """

    def __init__(self, num_heads, num_mod):
        super().__init__()
        self.num_heads = num_heads
        self.num_mod = num_mod
        ##MLP
        self.fc1 = nn.Linear(53, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        
        ##RESNET
        self.resnet18 = models.resnet18(pretrained=True)
        n_inputs = self.resnet18.fc.in_features

        self.resnet18.fc = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(n_inputs, 512))
        ])) 

        self.resnet_classification = nn.Linear(512, 2) 
        
        ##BERT
        self.bert = BertModel.from_pretrained('bert-base-uncased') 
        self.bert_classification = nn.Linear(768, 2)
        
        #Multi-modal classification layers
        self.multimodal_classification = nn.Linear(self.num_mod * 256, 2)
        self.pairwise_classification = nn.Linear(256*self.num_mod*(self.num_mod-1), 2)

        #projection to the same embedding size
        self.res_wrap = nn.Linear(512, 256)
        self.bert_wrap = nn.Linear(768, 256)

        self.dropout = nn.Dropout(0.25)
        
        #attention schemes
        self.pairwise_attention  = nn.MultiheadAttention(256, self.num_heads, batch_first = True)
        self.self_attention  = nn.MultiheadAttention(self.num_mod * 256, self.num_heads, batch_first = True)
        self.OvO_multihead_attention = MultiHeadAttention(256,self.num_heads)

        
    def bi_directional_att(self, pair):
        x = pair[0]
        y = pair[1]
        attn_output_LV, attn_output_weights_LV = self.pairwise_attention(x, y, y)
        attn_output_VL, attn_output_weights_VL = self.pairwise_attention(y, x, x)
        combined = torch.cat((attn_output_LV,
                              attn_output_VL), dim=1)
        return combined

    def forward(self, x, model):
        """
        Passes the input through the specified model architecture and returns the output.

        Args:
            x (tuple, list, or torch.Tensor): Input data for the model.
                If `model` is `"bert"`, `x` should be a tuple of input tensors `(text, masks)`, where
                `text` is a 2D tensor of shape `(batch_size, max_seq_len)` containing token ids,
                and `masks` is a tensor of shape `(batch_size, max_seq_len)` containing attention masks.
                If `model` is `"resnet"`, `x` should be a tensor of shape `(batch_size, input_size)`.
                Otherwise, `x` should be a list of modality inputs shaped `(batch_size, input_size)` followed by the mask (if text is included).
            model (str): Name of the model architecture to use. Available options are:
                - `"mlp"`: Multi-layer perceptron
                - `"resnet"`: ResNet18
                - `"bert"`: Bert model
                - `"bert_resnet"`: Concatenation of ResNet18 and Bert models
                - `"bert_resnet_OvO"`: ResNet18 and Bert models with OvO attention
                - `"bert_resnet_cross"`: ResNet18 and Bert models with pairwise cross-modal attention
                - `"bert_resnet_self"`: ResNet18 and Bert models with early-fusion to self-attention
                - `"bert_resnet_mlp"`: Concatenation of ResNet18, Bert, and MLP
                - `"bert_resnet_mlp_cross"`: ResNet18, Bert, MLP, and pairwise cross-modal attention
                - `"bert_resnet_mlp_self"`: ResNet18, Bert, MLP, with early-fusion to self-attention
                - `"bert_resnet_mlp_OvO"`: ResNet18, Bert, MLP, with OvO attention
                
                
        Returns:
            torch.Tensor: Output tensor of the specified model architecture. The shape of the output tensor 
            depends on the model architecture.
        """
        if model == "mlp":
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            out = self.fc3(x)
            
        elif model == "resnet":
            res = self.resnet18(x)       
            out = self.resnet_classification(res)
        
        elif model == "bert":
            text, masks = x
            bert = self.bert(text, attention_mask=masks, token_type_ids=None).last_hidden_state[:,0,:]
            out = self.bert_classification(bert)
            
        elif model == "bert_resnet":
            img, text, masks = x
            res_emb = self.resnet18(img)
            res = self.res_wrap(res)
            bert_emb = self.bert(text, attention_mask=masks).last_hidden_state[:,0,:]
            bert = self.bert_wrap(bert)
            combined = torch.cat((res,
                                  bert), dim=1)
            out = self.multimodal_classification(combined)
        
        elif model == "bert_resnet_OvO":
            img, text, masks = x
            
            res = self.resnet18(img)
            res = self.res_wrap(res)
            bert = self.bert(text,attention_mask=masks).last_hidden_state[:,0,:]
            bert = self.bert_wrap(bert) 
            attn_output_LV = self.OvO_multihead_attention([res, res], bert)
            attn_output_VL = self.OvO_multihead_attention([bert, bert], res)

            combined = torch.cat((attn_output_LV.squeeze(1),
                                  attn_output_VL.squeeze(1)), dim=1)
            out = self.multimodal_classification(combined)
        
        elif model == "bert_resnet_cross": 
            img, text, masks = x
            res_emb = self.resnet18(img)
        
            bert_emb = self.bert(text,attention_mask=masks).last_hidden_state[:,0,:] 
            res = self.res_wrap(res_emb)
            bert = self.bert_wrap(bert_emb)

            attn_output_LV, attn_output_weights_LV = self.pairwise_attention(bert, res, res)
            attn_output_VL, attn_output_weights_VL = self.pairwise_attention(res, bert, bert)

            combined = torch.cat((attn_output_LV,
                                  attn_output_VL), dim=1)
            out = self.multimodal_classification(combined)

        elif model == "bert_resnet_self":   
            img, text, masks = x
            res_emb = self.resnet18(img)
        
            bert_emb = self.bert(text,attention_mask=masks).last_hidden_state[:,0,:] 
            res = self.res_wrap(res_emb)
            bert = self.bert_wrap(bert_emb)
            combined = torch.cat((bert, res), dim=1)

            attn_output, attn_output_weights = self.self_attention(combined, combined, combined)
            out = self.multimodal_classification(attn_output)
   
        elif model == "bert_resnet_mlp":
            features, img, text, masks = x
            
            feat = self.fc1(features)
            feat = self.relu(feat)
            feat = self.fc2(feat)
            
            bert = self.bert(text,attention_mask=masks).last_hidden_state[:,0,:] 
            bert = self.bert_wrap(bert)
            res = self.resnet18(img)
            res = self.res_wrap(res)

            combined = torch.cat((bert, feat, res), dim=1)
            out = self.multimodal_classification(combined)
        
        elif model == "bert_resnet_mlp_cross":
            features, img, text, masks = x
            
            feat = self.fc1(features)
            feat = self.relu(feat)
            feat = self.fc2(feat)
            
            bert = self.bert(text,attention_mask=masks).last_hidden_state[:,0,:]
            bert = self.bert_wrap(bert)
            
            res = self.resnet18(img)
            res = self.res_wrap(res)
            
            pairs = [[feat, bert],[feat,res],[bert,res]]
        
            results = []
            for pair in pairs:
                combined = self.bi_directional_att(pair)
                results.append(combined)

            comb = torch.cat(results, dim=1)
            out = self.pairwise_classification(comb)

        elif model == "bert_resnet_mlp_self":   
            features, img, text, masks = x
            
            feat = self.fc1(features)
            feat = self.relu(feat)
            feat = self.fc2(feat)
            
            bert = self.bert(text,attention_mask=masks).last_hidden_state[:,0,:]
            bert = self.bert_wrap(bert)
            
            res = self.resnet18(img)
            res = self.res_wrap(res)
            combined = torch.cat((bert, feat, res), dim=1)

            attn_output, attn_output_weights = self.self_attention(combined, combined, combined)
            out = self.multimodal_classification(attn_output)    
        else:
            features, img, text, masks = x
            
            feat = self.fc1(features)
            feat = self.relu(feat)
            feat = self.fc2(feat) 
            
            bert = self.bert(text,attention_mask=masks).last_hidden_state[:,0,:] 
            bert = self.bert_wrap(bert) 
            
            res = self.resnet18(img)
            res = self.res_wrap(res) 
            
            attn_txt = self.OvO_multihead_attention([feat, res], bert) 
            attn_img = self.OvO_multihead_attention([feat, bert],res)
            attn_tab = self.OvO_multihead_attention([bert, res], feat)

            comb = torch.cat([attn_txt.squeeze(1), attn_img.squeeze(1), attn_tab.squeeze(1)], dim=1)
            out = self.multimodal_classification(comb)

        return out

