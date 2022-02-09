import re
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_packed import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
import sys,logging
import dill
from transformers import AutoTokenizer, AutoModel

class mllt(nn.Module):
    def __init__(self,class_3):
        super(mllt, self).__init__()
        self.hidden_size = 768
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        if class_3:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 3),
                )
        else:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 25),
                )
            
        self.fc_key = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_query = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_value = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.drop_out = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        # self.VisitGRU =  nn.GRU(input_size= self.hidden_size, batch_first=True, hidden_size= self.hidden_size, num_layers=1, bidirectional=False)
        self.VisitGRU =  nn.LSTM(input_size= self.hidden_size//2, batch_first=True, hidden_size= self.hidden_size//2, num_layers=1, bidirectional=False)


   

    # def encoder(self, input_ids=None,
    #         attention_mask=None,
    #         head_mask=None,
    #         encoder_outputs=None,
    #         inputs_embeds=None,
    #         output_attentions=None,
    #         output_hidden_states=None,
    #         return_dict=None):

    #     encoder_outputs = self.bart.encoder(
    #     input_ids=input_ids,
    #     attention_mask=attention_mask,
    #     head_mask=head_mask,
    #     inputs_embeds=inputs_embeds,
    #     output_attentions=output_attentions,
    #     output_hidden_states=output_hidden_states,
    #     return_dict=return_dict)
    #     # print(encoder_outputs[0][:,[0],:5])
    #     return encoder_outputs


    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        # print("v :", v)
        # print("c :", c)

        v = self.drop_out(self.fc_key(v))
        c = self.drop_out(self.fc_query(c))
        g = torch.bmm(v, c.transpose(-2, -1))

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]

        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print("b: ",b[[1],:,:].squeeze().squeeze())
        return b   
         
        
   

    def approximation(self,label_embedding, Ot,self_att):
        Ot_E_batch = self.encoder(**Ot).last_hidden_state
        if self_att == "self_att":
            attention_weights =  self.cross_attention(Ot_E_batch,Ot_E_batch)
            Ot_E_batch = self.drop_out(self.fc_value(Ot_E_batch))
            Ztd =   self.drop_out(Ot_E_batch * attention_weights).sum(1)
            return Ztd        
        elif self_att == "cross_att":
            attention_weights = self.cross_attention(Ot_E_batch,label_embedding.unsqueeze(0).repeat(Ot_E_batch.shape[0],1,1))
            Ot_E_batch = self.drop_out(self.fc_value(Ot_E_batch))
            Ztd =   self.drop_out(Ot_E_batch * attention_weights).sum(1)
            return Ztd
 
    def emission(self,Ztd):
        ### embedding (seq embedding) 
        ### check 原始 embedding size in decoderduide
        Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))

        return Yt

    def transition_rnn(self,Ztd):
        all_h,last_h = self.VisitGRU(Ztd)
        # all_h = torch.mean(all_h.view(all_h.shape[0],all_h.shape[1],2, all_h.shape[-1]//2),-2)
        return all_h


    def forward(self,label_token,Ot,self_att):
        # print(Ztd_last.shape)
        Ztd = self.approximation(label_token, Ot,self_att)
        return Ztd
        # return  Ot_,Yt,Ot
           

   



