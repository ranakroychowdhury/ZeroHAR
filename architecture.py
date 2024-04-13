# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 15:04:31 2020

@author: Ranak Roy Chowdhury
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import transformer
# torch.autograd.set_detect_anomaly(True)


class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model, dropout = 0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[: , 0 : -1]
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    # Input: seq_len x batch_size x dim, Output: seq_len, batch_size, dim
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    

class Permute021(torch.nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)



class Permute201(torch.nn.Module):
    def forward(self, x):
        return x.permute(2, 0, 1)



class Permute120(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 2, 0)



class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim = -1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp



class GZSLModel(nn.Module):

    def __init__(self, stage, device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nlayers, pooling, classid_to_word_embedding, sensor_description_to_word_embedding, dropout = 0.1):
        super(GZSLModel, self).__init__()
        # from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        self.stage = stage

        self.nclasses = nclasses

        self.trunk_net = nn.Sequential(
            nn.Linear(input_size, emb_size), # batch, seq, emb = batch, seq, input
            Permute021(), # batch, emb, seq = batch, seq, emb 
            nn.BatchNorm1d(emb_size), # batch, emb, seq = batch, emb, seq
            Permute201(), # seq, batch, emb = batch, emb, seq
            PositionalEncoding(seq_len, emb_size, dropout), # seq, batch, emb = seq, batch, emb
            Permute120(), # batch, emb, seq = seq, batch, emb
            nn.BatchNorm1d(emb_size) # batch, emb, seq = batch, emb, seq
        )
        
        # encoder_layers = transformer_encoder_class.TransformerEncoderLayer(emb_size, nhead, nhid, out_channel, filter_height, filter_width, dropout)
        # encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        encoder_layers = transformer.TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = transformer.TransformerEncoder(encoder_layers, nlayers, device)
        
        self.pooling = pooling

        self.batch_norm = nn.BatchNorm1d(emb_size)
    
        self.cls_emb = nn.Embedding(1, emb_size)

        self.sim = Similarity(temp = 0.05)

        # Classification Layers
        self.class_description_to_word_embedding = [classid_to_word_embedding[classid] for classid in range(self.nclasses)]
        self.class_description_to_word_embedding = torch.cat(self.class_description_to_word_embedding, axis = 0) # nclasses * 10 (number of descriptions per class), 1024
        self.sensor_description_to_word_embedding = sensor_description_to_word_embedding # input, 1024
        
        print('sensor_description_to_word_embedding shape: ' + str(self.sensor_description_to_word_embedding.shape))
        print('class_description_to_word_embedding shape: ' + str(self.class_description_to_word_embedding.shape))


        #-------USE self.downsample_embedding_to_input_net AND self.sequence_representation_net JUST FOR STAGE 1-------#
        self.downsample_embedding_to_input_net = nn.Sequential(
                                            Permute021(), # batch, seq, emb = batch, emb, seq
                                            nn.Linear(emb_size, input_size), # batch, seq, input = batch, seq, emb. 64 -> 45, 27, 6, 6, 6 input_size for the 5 datasets
                                            nn.ReLU(),
                                            Permute021(), # batch, input, seq = batch, seq, input
                                            nn.BatchNorm1d(input_size), # batch, input, seq = batch, input, seq
                                            nn.Dropout(p = 0.3)
                                        )

        self.sequence_representation_net = nn.Sequential(
                                        nn.Linear(seq_len, nhid), # batch, input, nhid = batch, input, seq                                
                                        nn.ReLU(),
                                        Permute021(), # batch, nhid, input = batch, input, nhid
                                        nn.BatchNorm1d(nhid), # batch, nhid, input = batch, nhid, input
                                        Permute021(), # batch, input, nhid = batch, nhid, input
                                        nn.Dropout(p = 0.3)
        )
        #-------USE self.downsample_embedding_to_input_net AND self.sequence_representation_net JUST FOR STAGE 1-------#


        #-------USE self.imu_to_shared_net JUST FOR STAGE 2----------#
        self.imu_to_shared_net = nn.Sequential(
                            nn.Linear(emb_size, nhid), # batch, nhid = batch, emb
                            nn.ReLU(),
                            nn.BatchNorm1d(nhid), # batch, nhid = batch, nhid
                            nn.Dropout(p = 0.3),
                            nn.Linear(nhid, nhid), # batch, nhid = batch, nhid
                            nn.ReLU(),
                            nn.BatchNorm1d(nhid), # batch, nhid = batch, nhid
                            nn.Dropout(p = 0.3)
                        )
        #-------USE self.imu_to_shared_net JUST FOR STAGE 2----------#


        #-------USE self.text_to_shared_net FOR STAGE 1 AND STAGE 2----------#
        self.text_to_shared_net = nn.Sequential(
                            nn.Linear(1024, nhid),
                            nn.ReLU(),
                            nn.LayerNorm(nhid),
                            nn.Dropout(p = 0.3)
                        )
        #-------USE self.text_to_shared_net FOR STAGE 1 AND STAGE 2----------#

        

    def forward(self, x):

        batch_size, seq_len, input_size = x.shape # batch, seq, input
        x = self.trunk_net(x) # batch, emb, seq = batch, seq, input
        x = x.permute(2, 0, 1) # seq, batch, emb = batch, emb, seq
        
        # create CLS token representation
        if self.pooling == 'att' or self.pooling == 'bert':
            cls_tokens = torch.zeros(batch_size).to(x.device).long() # batch
            cls_repr = self.cls_emb(cls_tokens).view(1, batch_size, -1)  # 1, batch, emb

        # append CLS token at the beginning of input
        if self.pooling == 'bert': x = torch.cat([cls_repr, x], dim=0) # seq + 1, batch, emb

        x, attn = self.transformer_encoder(x) # seq, batch, emb = seq, batch, emb
        x = self.batch_norm(x.permute(1, 2, 0)) # batch, emb, seq = seq, batch, emb

        if self.stage == '1':
            x = self.downsample_embedding_to_input_net(x) # batch, input, seq = batch, emb, seq
            imu = self.sequence_representation_net(x) # batch, input, nhid = batch, input, seq

            sensors = self.text_to_shared_net(self.sensor_description_to_word_embedding.to(x.device)) # input, nhid = input, 1024
            sensors = sensors.unsqueeze(0).repeat(batch_size, 1, 1) # batch, input, nhid = input, nhid

            imu_expanded = imu.unsqueeze(2).expand(-1, -1, input_size, -1) # batch, input, input, nhid = batch, input, nhid
            sensors_expanded = sensors.unsqueeze(1).expand(-1, input_size, -1, -1) # batch, input, input, nhid = batch, input, nhid

            cos_sim_imu_to_sensor = self.sim(imu_expanded, sensors_expanded) # batch, input, input

            return cos_sim_imu_to_sensor

        else:
            # process the transformer output sepending on pooling type (how to compress data along the sequence length axis: seq, batch, emb -> batch, nhid
            if self.pooling == 'bert': 
                y_feat = x[ : , : , 0] # batch, emb
            elif self.pooling == 'att': 
                weights = F.softmax(torch.bmm(cls_repr.permute(1, 0, 2), x), dim = -1) # (batch, 1, emb) x (batch, emb, seq) = (batch, 1, seq)
                y_feat = torch.sum(x * weights, dim=-1) # batch, emb = (batch, emb, seq) x (batch, 1, seq)
            elif self.pooling == 'mean': 
                y_feat = torch.mean(x, axis = 0) # batch, emb
            else:
                y_feat = x[ : , : , -1] # batch, emb

            imu_to_shared_embedding = self.imu_to_shared_net(y_feat) # batch, nhid = batch, emb

            self.trained_text_to_shared_embedding = self.text_to_shared_net(self.class_description_to_word_embedding.to(x.device)) # (nclasses * 10 - number of descriptions per class, nhid) = (nclasses * 10 - number of descriptions per class, 1024)
            reshaped_tensor = self.trained_text_to_shared_embedding.view(self.nclasses, -1, self.trained_text_to_shared_embedding.shape[1]) # (nclasses, 10, nhid)
            self.mean_trained_text_to_shared_embedding = reshaped_tensor.mean(dim = 1) # (nclasses, nhid)
            
            y = torch.matmul(imu_to_shared_embedding, self.mean_trained_text_to_shared_embedding.T.to(x.device)) # (batch, nclasses) = (batch, nid) x (nhid, nclasses)
            
            return y, attn
        

        
        
        



'''
device = 'cuda:1'    
lr, dropout = 0.01, 0.01
nclasses, seq_len, batch, input_size = 12, 5, 11, 10
emb_size, nhid, nhead, nlayers = 32, 128, 2, 3

classid_to_word_embedding = {}
for i in range(nclasses):
    classid_to_word_embedding[i] = torch.randn(nhid)
pooling = 'anything'

model = GZSLModel(device, nclasses, seq_len, batch, input_size, emb_size, nhead, nhid, nlayers, pooling, classid_to_word_embedding, dropout = 0.1).to(device)

x = torch.randn(batch, seq_len, input_size) * 50
x = torch.as_tensor(x).float()
print("x: " + str(x.shape))

out, attn = model(torch.as_tensor(x, device = device))
# print(attn.shape)
print("out: " + str(out.shape))
'''

# print(model)


# =============================================================================
# print(sum([param.nelement() for param in model.parameters()]))
# print(summary(model(), torch.zeros((seq_len, input_size)), show_input=False))
# summary(model, (1, seq_len, input_size))
# children, a = 0, []
# for child in model.children():
#     print('Child: ' + str(children))
#     children += 1
#     paramit = 0
#     for param in child.parameters():
#         a.append(param.data)
#         print('Paramit: ' + str(paramit))
#         paramit += 1    
# =============================================================================