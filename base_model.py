import re
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_packed import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from transformers import AutoTokenizer, AutoModel

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
   
class mllt(nn.Module):
    def __init__(self, ):
        super(mllt, self).__init__()
        self.hidden_size = 768
        self.bart  = BartModel.from_pretrained('facebook/bart-base')
        self.encoder1 = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        self.fc = nn.Linear(self.hidden_size,300)

        self.MLPs = nn.Sequential(
            nn.Linear(self.hidden_size, 25),
            )

        self.phrase_filter = nn.Conv2d(
            # dilation= 2,
            in_channels=1,
            out_channels=1,
            padding='same',
            kernel_size=(3,1))
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
        v = self.fc(v)
        c = self.fc(c)
        g = torch.bmm(v, c.transpose(-2, -1))
        u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        print(": ",b[:1,:,:].squeeze().squeeze())

        return b   
        
    
    def padding_batch(self,vec,batch_output):
        # max_length = max([i.shape[1] for i in vec])
        max_length = max([i["input_ids"].shape[0] for i in batch_output])
        padding_batch_list = []
        for t in vec:
            if t.shape[1] < max_length:
                padding = torch.zeros((1,max_length-t.shape[1],768)).to(f"cuda:{t.get_device()}")
                t = torch.cat([t,padding],dim=1)
            # print(max_length,t.shape)
            padding_batch_list.append(t)
        embedded = torch.cat(padding_batch_list,dim=1)
        return embedded


    def approximation(self,At, Ot):
        Ot_E_batch = self.encoder1(**Ot).last_hidden_state
        if len(At)> 2:
            # for A in At:
            At_E_batch = torch.cat([self.encoder1(**A).last_hidden_state.sum(1).unsqueeze(0) for A in At],dim=0)

        else:
            At_E_batch = self.encoder1(**At).last_hidden_state.sum(1).unsqueeze(0)
        Ztd = self.drop_out(Ot_E_batch * self.cross_attention(Ot_E_batch,At_E_batch) ).sum(1)
        return Ztd,Ot_E_batch
 
    def emission(self,Ztd):
        ### embedding (seq embedding) 
        ### check 原始 embedding size in decoderduide
        Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))

        return Yt



    def forward(self,At,Ot):
        # print(Ztd_last.shape)
        Ztd,Ot_E_batch = self.approximation(At, Ot)
        Yt = self.emission(Ztd)
        return Yt
        # return  Ot_,Yt,Ot
           

   



