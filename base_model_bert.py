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
from transformers import AutoModel

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
        self.encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

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

   

   

    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        g = torch.bmm(v, c.transpose(-2, -1))

        u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]

        m = self.drop_out(F.max_pool2d(u,kernel_size = (1,u.shape[-1]))).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        print("b: ",b[:1,:,:].squeeze().squeeze())
        return b   
        


    def forward(self,At,Ot):
         
        At_E_batch = self.encoder(**At)[0][:,[0],:].transpose(1,0)
        Ot_E_batch = self.encoder(**Ot)[0]
        Ztd = self.drop_out(Ot_E_batch[:,1:-1,:] * self.cross_attention(Ot_E_batch[:,1:-1,:],At_E_batch)).sum(1)        
        Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))
        return Yt
           

   



