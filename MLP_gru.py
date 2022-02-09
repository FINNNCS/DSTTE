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

class MLP_RNN(nn.Module):
    def __init__(self, num_embeddings):
        super(MLP_RNN, self).__init__()
        self.hidden_size = 768
        self.word_embedding = nn.Embedding(num_embeddings, self.hidden_size, padding_idx=0) ## for random c
        self.VisitGRU =  nn.GRU(input_size= self.hidden_size, batch_first=True, hidden_size= self.hidden_size, num_layers=1, bidirectional=False)
        self.fc = nn.Linear(self.hidden_size,300)

        self.MLPs = nn.Sequential(
            nn.Linear(self.hidden_size, 25),
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
        # v =  self.drop_out(self.fc(v))
        # c = self.drop_out(self.fc(c))
        # print(v.shape,c.shape)

        v =  self.fc(v)
        c = self.fc(c)

        g = torch.bmm(v, c.transpose(-2, -1))
        # print(g.shape)
        # u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print(": ",b[:1,:,:].squeeze().squeeze())

        return b   
        
    def transition_rnn(self,Ztd):
        all_h,last_h = self.VisitGRU(Ztd)
        # all_h = torch.mean(all_h.view(all_h.shape[0],all_h.shape[1],2, all_h.shape[-1]//2),-2)
        return self.drop_out(all_h)
 
    def emission(self,Ztd):
        ### embedding (seq embedding) 
        ### check 原始 embedding size in decoderduide
        Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))

        return Yt
    def forward(self,Ot,At,label_embedding):
        Ot = self.drop_out(self.word_embedding(Ot))  # [b, l, h]
        At = self.drop_out(self.word_embedding(At))  # [b, l, h]

        Lt = self.drop_out(self.word_embedding(label_embedding)).sum(1).unsqueeze(0).detach()  # [b, l, h]
        # print(Lt)
        Ztd = self.drop_out(self.cross_attention(Ot,Lt)*Ot).sum(1)
        return Ztd
        # return  Ot_,Yt,Ot
           

   



