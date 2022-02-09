import re
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_packed import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, AutoModel
from torch.distributions.studentT import StudentT
from tqdm import tqdm
import numpy as np
from transformers import BartModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
import sys,logging
logging.root.handlers = []
logging.basicConfig(level="INFO", format = '%(asctime)s:%(levelname)s: %(message)s' ,stream = sys.stdout)
logger = logging.getLogger(__name__)
logger.info('hello')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)

def check_memory():
    logger.info('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))
   
class encoder_net(nn.Module):
    def __init__(self,class_3):
        super(encoder_net, self).__init__()
        self.hidden_size = 768
        self.bart  = BartModel.from_pretrained('facebook/bart-base')
        # self.encoder1 = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.fc = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.fuse_fc = nn.Linear(self.hidden_size,self.hidden_size//2)
        self.selector_fc = nn.Linear(self.hidden_size,self.hidden_size//2)
        if class_3:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 3),
                )
        else:
            self.MLPs = nn.Sequential(
                nn.Linear(self.hidden_size//2, 25),
                )
            
        self.drop_out = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
   

    def encoder(self, input_ids=None,
            attention_mask=None,
            head_mask=None,
            encoder_outputs=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        encoder_outputs = self.bart.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict)
        # print(encoder_outputs[0][:,[0],:5])
        return encoder_outputs


    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        v = self.drop_out(self.fc(v))
        c = self.drop_out(self.fc(c))
        g = torch.bmm(v, c.transpose(-2, -1))
    
        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print(": ",b[:1,:,:].squeeze().squeeze())

        return b   
        
   
    def approximation(self, Ot,label_token,self_att):
        Ot_E_batch = self.encoder(**Ot).last_hidden_state
        label_embedding = self.encoder(**label_token).last_hidden_state.sum(1).detach()
        if self_att == "self_att":
            attention_weights =  self.cross_attention(Ot_E_batch,Ot_E_batch)
            Ot_E_batch = self.drop_out(self.fc(Ot_E_batch))
            Ztd =   self.drop_out(Ot_E_batch * attention_weights).sum(1)
            return Ztd        
        elif self_att == "cross_att":
            attention_weights = self.cross_attention(Ot_E_batch,label_embedding.unsqueeze(0).repeat(Ot_E_batch.shape[0],1,1))
            Ot_E_batch = self.drop_out(self.fc(Ot_E_batch))
            Ztd =   self.drop_out(Ot_E_batch * attention_weights).sum(1)
            return Ztd   
        else:
            return Ot_E_batch.sum(1) 
 

    def forward(self,Ot,label_embedding,self_att):
        Ztd = self.approximation(Ot,label_embedding,self_att)
        Yt = self.sigmoid(self.drop_out(self.MLPs(Ztd)))

        return Yt,Ztd
           

    
class cluster_net(nn.Module):
    def __init__(self):
        super(cluster_net, self).__init__()
        self.hidden_size = 768
        self.alpha = 1
        self.cluster_centers = 10

        self.cluster_fc = nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size//2),
                    nn.LeakyReLU(inplace = True)
                    )
        # self.MLPs = nn.Sequential(
        #     nn.Linear(self.hidden_size//2, 25),
        #     )
            
        self.drop_out = nn.Dropout(0.3)
        # self.sigmoid = nn.Sigmoid()
    def get_cluster_prob(self, embeddings,Center):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - Center) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def target_distribution(self,batch):
        weight = (batch ** 2) / (torch.sum(batch, 0) + 1e-9)
        return (weight.t() / torch.sum(weight, 1)).t()
    def selector(self,Center,phi):
        index_list = torch.max(phi,dim = -1).indices
        
        tmp = []
        # print("st: ",index_list)

        for i in index_list:
            selected_tensor = Center[[i],:]
            tmp.append(selected_tensor)
        return torch.cat(tmp,0)

    def forward(self,Ztd,M,weighted):
        # print("Center: ",M.grad)
        Center = self.drop_out(self.cluster_fc(M)/384)
        # print("Center: ",torch.max(Center[[0],:]),torch.min(Center[[0],:]))
        # print("Ztd: ",torch.max(Ztd[[0],:]),torch.min(Ztd[[0],:]))
        # p = StudentT(F.softmax(Ztd))
        # print(Center)

        phi = self.get_cluster_prob(Ztd,Center)#phi batch x 10 x 1
        print("phi: ",phi[[0],:])
        print(phi.requires_grad)

        cluster_target =self.target_distribution(phi).detach()
        if weighted:     
            ct = self.drop_out(Center * phi.unsqueeze(-1)).sum(1)
        else:
            ct = self.selector(Center,phi)

        return ct,phi,cluster_target


