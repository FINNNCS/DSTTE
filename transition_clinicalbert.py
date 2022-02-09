import re
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_packed import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
from collections import deque
import torch.optim as optim

class mllt(nn.Module):
    def __init__(self,Add_additions,visit):
        super(mllt, self).__init__()
        self.visit = visit
        self.hidden_size = 768
        self.last_patient_id = deque([0])
        self.drop_out = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_size,300)

        if Add_additions:
            self.transdplus = nn.GRU(input_size= self.hidden_size*3, batch_first=True, hidden_size= self.hidden_size, num_layers=1, bidirectional=True)
        else:
            self.transd = nn.GRU(input_size= self.hidden_size, batch_first=True, hidden_size= self.hidden_size, num_layers=1, bidirectional=True)

        self.transd_mean = nn.Linear( self.hidden_size,  self.hidden_size)
        self.transd_logvar = nn.Linear( self.hidden_size,  self.hidden_size)

        self.transRNN =  nn.GRU(input_size= self.hidden_size, batch_first=True, hidden_size= self.hidden_size, num_layers=1, bidirectional=True)

        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.zd_mean = nn.Linear( self.hidden_size,  self.hidden_size)
        self.zd_logvar = nn.Linear( self.hidden_size,  self.hidden_size)  
        self.Ztd_cat = nn.Linear(self.hidden_size*2, self.hidden_size)

      
        self.forget_gate =  nn.Sequential(
            nn.Linear(self.hidden_size*2, self.hidden_size*2),
            nn.Dropout(0.3),
            nn.Sigmoid(),
            )


        self.MLPs = nn.Sequential(
            nn.Linear(self.hidden_size, 25),
            )


        self.fusion_fc = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.phrase_filter = nn.Conv2d(
            # dilation= 2,
            in_channels=1,
            out_channels=1,
            padding='same',
            kernel_size=(3,1))
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()


    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        v =  self.drop_out(self.fc(v))
        c = self.drop_out(self.fc(c))
        # v =  self.fc(v)
        # c = self.fc(c)
        g = torch.bmm(v, c.transpose(-2, -1))
        # print(g.shape)
        # u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print(": ",b[:1,:,:].squeeze().squeeze())

        return b   
        

    def sampling(self,mu,logvar,flag):
        if flag == "test":
            return mu
        std = torch.exp(0.5 * logvar).detach()        
        epsilon = torch.randn_like(std).detach()
        zt = epsilon * std + mu 
        return zt
        

    def approximation(self,Ztd_list,At, Ot,label_token,flag):
          
        # At_E_batch = self.encoder(**At)[0][:,[0],:].transpose(1,0)
        Ot_E_batch = self.encoder(**Ot).last_hidden_state
        label_embedding = self.encoder(**label_token).last_hidden_state.sum(1).detach()
        # print(label_embedding.shape)
        attention_weights = self.cross_attention(Ot_E_batch,label_embedding.unsqueeze(0))
        Ot_E_batch_cross_attention =   self.drop_out(Ot_E_batch * attention_weights ).sum(1)
        # Ot_E_batch_cross_attention = self.drop_out(Ot_E_batch[:,1:-1,:] * self.cross_attention(Ot_E_batch[:,1:-1,:],At_E_batch)).sum(1)

        # if self.visit == "once":
        #     return Ot_E_batch_cross_attention,Ot_E_batch_cross_attention,Ot_E_batch_cross_attention
        # if len(Ztd_list.shape) < 3:
        #     Ztd_list = Ztd_list.unsqueeze(1)
        _,Ztd_last = self.transRNN(Ztd_list.unsqueeze(0))
        Ztd_last =  self.drop_out(torch.mean(Ztd_last,0))


        # Ztd = torch.cat((Ztd_last,(Ot_E_batch*torch.tensor([0.0033]*300).unsqueeze(0).unsqueeze(-1).to(f"cuda:{Ztd_last.get_device()}")).sum(1)),-1)
        Ztd = torch.cat((Ztd_last,Ot_E_batch_cross_attention),-1)

        # Ztd = torch.cat((Ztd_list[-1,:].unsqueeze(0),Ot_E_batch_cross_attention),-1)
        gate_ratio_ztd = self.forget_gate(Ztd)

        Ztd = self.drop_out( self.Ztd_cat(gate_ratio_ztd*Ztd))
        Ztd_mean = self.zd_mean(Ztd)
        Ztd_logvar = self.zd_logvar(Ztd)

        Ztd_s = self.sampling(Ztd_mean,Ztd_logvar,flag)
      
        return Ztd_s,Ztd_mean,Ztd_logvar,attention_weights
 
    def emission(self,Ztd,At):
        ### embedding (seq embedding) 
        ### check 原始 embedding size in decoder
        Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))

        return Yt

    def trasition(self,Ztd_last,chief_comp_last,Ts,Add_additions):
        if Add_additions:
            Ct_E = self.encoder(**chief_comp_last[-1])[0].sum(0).sum(0)        
            if len(chief_comp_last) > 1:
                Ct_E_last = self.encoder(**chief_comp_last[-2])[0].sum(0).sum(0)      
            else:
                Ct_E_last = Ct_E
            Ct_diff = Ct_E - Ct_E_last

            _,Ztd_last_last_hidden = self.transdplus(torch.cat((Ztd_last.unsqueeze(0),Ct_diff.unsqueeze(0).unsqueeze(0),Ts.unsqueeze(0).unsqueeze(0).repeat(1,1,768)),-1))
            Ztd =  self.drop_out(torch.mean(Ztd_last_last_hidden,0))
            Ztd_mean =   self.drop_out(self.transd_mean(Ztd))
            Ztd_logvar =  self.drop_out(self.transd_logvar(Ztd))
            return Ztd_mean,Ztd_logvar

        _,Ztd_last_last_hidden = self.transd(Ztd_last.unsqueeze(0))

        Ztd =  self.drop_out(torch.mean(Ztd_last_last_hidden,0))

        Ztd_mean =   self.drop_out(self.transd_mean(Ztd))
        Ztd_logvar =  self.drop_out(self.transd_logvar(Ztd))

        return Ztd_mean,Ztd_logvar


    def forward(self,Ztd_list,label_embedding,Ts,Ct,At,Ot,Ztd_last,Add_additions,flag):
        Ztd_list = torch.cat(Ztd_list,0).to(Ztd_last.device)

        Ztd,Ztd_mean_post,Ztd_logvar_post,attention_weights = self.approximation(Ztd_list,At, Ot,label_embedding,flag)
        Yt = self.emission(Ztd,At)
        Ztd_mean_priori,Ztd_logvar_priori = self.trasition(Ztd_last,Ct,Ts,Add_additions)
        return Ztd,Ztd_mean_post,Ztd_logvar_post,Yt,Ztd_mean_priori,Ztd_logvar_priori,attention_weights
        # return  Ot_,Yt,Ot
           

   

   