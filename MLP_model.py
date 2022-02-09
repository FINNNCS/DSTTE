import re
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_packed import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
import sys,logging

class MLP(nn.Module):
    def __init__(self, class_3, num_embeddings):
        super(MLP, self).__init__()
        self.hidden_size = 768
        self.word_embedding = nn.Embedding(num_embeddings, self.hidden_size, padding_idx=0) ## for random c
        self.fc_key = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_query = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fc_value = nn.Linear(self.hidden_size,self.hidden_size//2)
        if class_3:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 3),
                )
        else:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 25),
                )
        self.drop_out = nn.Dropout(0.3)
        self.phrase_filter = nn.Conv2d(
            # dilation= 2,
            in_channels=1,
            out_channels=1,
            padding='same',
            kernel_size=(3,1))
        self.sigmoid = nn.Sigmoid()

   


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
        # print("b: ",b.squeeze().squeeze(),torch.max(b),torch.sum(b))
        return b  
        
        


    def forward(self,Ot,label_embedding,self_att):
        # print(Ztd_last.shape)
        Ot = self.drop_out(self.word_embedding(Ot))  # [b, l, h]
        # At = self.drop_out(self.word_embedding(At))  # [b, l, h]
        Lt = self.drop_out(self.word_embedding(label_embedding))  # [b, l, h]
        if self_att == "self_att":
            attention_weights =  self.cross_attention(Ot,Ot)
            Ztd =   (self.drop_out(self.drop_out(self.fc_value(Ot))) * attention_weights).sum(1)
            Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))
            return Yt
        elif self_att == 'cross_att':
            # print(Ot.shape,Lt.shape)
            attention_weights = self.cross_attention(Ot,Lt)

            Ztd =   (self.drop_out(self.drop_out(self.fc_value(Ot))) * attention_weights).mean(1)
            Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))
            return Yt
        else:
            Yt =  self.sigmoid(self.drop_out(self.MLPs(self.fc_value(Ot).mean(1))))
            return Yt
           

   



