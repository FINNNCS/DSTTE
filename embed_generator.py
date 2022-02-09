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
        # self.encoder1 = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.drop_out = nn.Dropout(0.3)

        self.fc = nn.Linear(self.hidden_size,384)

    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        v = self.fc(v)
        c = self.fc(c)
        g = torch.bmm(v, c.transpose(-2, -1))
        # print(g.shape)
        # u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print(": ",b[:1,:,:].squeeze().squeeze())

        return b   
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




    def forward(self,Ot,label_token):
        # print(Ztd_last.shape)
        Ot_E_batch = self.encoder(**Ot).last_hidden_state
        label_embedding = self.encoder(**label_token).last_hidden_state.mean(1).detach()
        attention_weights = self.cross_attention(Ot_E_batch,label_embedding.unsqueeze(0).repeat(Ot_E_batch.shape[0],1,1))
        Ot_E_batch =   self.drop_out(Ot_E_batch * attention_weights).mean(1)
        # Ot_E_batch = self.encoder(**Ot).last_hidden_state.sum(1) 
        return Ot_E_batch
           

   



