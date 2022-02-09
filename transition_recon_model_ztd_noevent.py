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
logging.root.handlers = []
logging.basicConfig(level="INFO", format = '%(asctime)s:%(levelname)s: %(message)s' ,stream = sys.stdout)
logger = logging.getLogger(__name__)
logger.info('hello')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)

def check_memory():
    logger.info('GPU memory: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))
   
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

        self.bart  = BartModel.from_pretrained('facebook/bart-base')
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
        self.lm_head = nn.Linear(self.hidden_size, self.bart.shared.num_embeddings, bias=False)

        self.phrase_filter = nn.Conv2d(
            # dilation= 2,
            in_channels=1,
            out_channels=1,
            padding='same',
            kernel_size=(3,1))
        self.dropout = nn.Dropout(0.3)
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

        
    def decoder(self,encoder_output,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=True,
            output_hidden_states=None,
            return_dict=None,):
             # input_ids if no decoder_input_ids are provided
    
        output_attentions = output_attentions if output_attentions is not None else self.bart.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bart.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.bart.config.use_cache
        return_dict = return_dict if return_dict is not None else self.bart.config.use_return_dict
        decoder_outputs = self.bart.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # last_hidden_state = decoder_outputs.last_hidden_state
        # print("attentions: ",decoder_outputs[0][:1,:2,:])
        # print(last_hidden_state.shape, self.bart.shared.weight.T.shape)
        # seq_output = torch.matmul(last_hidden_state, self.bart.shared.weight.T)
        # seq_output = self.lm_head(last_hidden_state)
        # (self_attn_weights, cross_attn_weights)
        attantions = decoder_outputs.attentions[0]
        # attantions = torch.sum(attantions,1)/attantions.shape[1]
        attantions = attantions[:,-1,:,:].squeeze(1)
        ############## check attention 位置 match 不match， 是否要给cls空位置
        attantions = torch.softmax(torch.sum(attantions,1),dim = -1).squeeze(1)

        lm_logits = self.lm_head(decoder_outputs[0])
        return lm_logits
    def cross_attention(self,v,c):
        B, Nt, E = v.shape
        v = v / math.sqrt(E)
        v =  self.drop_out(self.fc(v))
        c = self.drop_out(self.fc(c))
        g = torch.bmm(v, c.transpose(-2, -1))
        # print(g.shape)
        # u = torch.relu(self.phrase_filter(g.unsqueeze(1)).squeeze(1))  # [b, l, k]

        m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
        b = torch.softmax(m, dim=1)  # [b, l, 1]
        # print(": ",b[:1,:,:].squeeze().squeeze())

        return b   
        
    def reshape_event(self,event):
        event_input_ids = event['input_ids']
        id_temp = []
        im_temp = []
        # print("before: ",len(event['input_ids']))
        ##  (number of event, sub-tokens)
        # print(event_input_ids)
        for i in range(event_input_ids.shape[0]):
            event_id = event_input_ids[i]
            for j in event_id:
                if i == 0 and j == 0:
                    continue

                elif i > 0 and j == 0:
                    continue
                elif i < event_input_ids.shape[0]-1 and j == 2:
                    continue
                elif i == event_input_ids.shape[0]-1 and j ==2:
                    id_temp.append(j.unsqueeze(0))
                    im_temp.append(torch.ones(1,dtype = torch.long).to(event_input_ids.device))
                elif j == 1:
                    continue 
                elif j != 1:
                    id_temp.append(j.unsqueeze(0))
                    im_temp.append(torch.ones(1,dtype = torch.long).to(event_input_ids.device))
        event['input_ids'] = torch.cat(id_temp).unsqueeze(0)
        event['attention_mask'] = torch.cat(im_temp).unsqueeze(0)
        # print("later: ",len(event['input_ids']))

        return event
     
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

        Ot_E_batch_cross_attention =   self.drop_out(Ot_E_batch * self.cross_attention(Ot_E_batch,label_embedding.unsqueeze(0)) ).sum(1)

        # Ot_E_batch_cross_attention = self.drop_out(Ot_E_batch[:,1:-1,:] * self.cross_attention(Ot_E_batch[:,1:-1,:],At_E_batch)).sum(1)
        # if self.visit == "once":
        #     return Ot_E_batch_cross_attention,Ot_E_batch_cross_attention,Ot_E_batch_cross_attention
        
        _,Ztd_last = self.transRNN(Ztd_list.unsqueeze(0))
        # Ztd_last = Ztd_last.view(Ztd_last.shape[0],Ztd_last.shape[1],2, Ztd_last.shape[-1]//2).sum(2).squeeze(1)
        Ztd_last =  self.drop_out(torch.mean(Ztd_last,0))

        # print(Ot_E_batch.shape,torch.tensor([0.0033]*300).unsqueeze(0).unsqueeze(-1).shape)
        # Ztd = torch.cat((Ztd_last,(Ot_E_batch*torch.tensor([0.0033]*300).unsqueeze(0).unsqueeze(-1).to(f"cuda:{Ztd_last.get_device()}")).sum(1)),-1)

        Ztd = torch.cat((Ztd_last,Ot_E_batch_cross_attention),-1)
        gate_ratio_ztd = self.forget_gate(Ztd)

        Ztd = self.drop_out( self.Ztd_cat(gate_ratio_ztd*Ztd))
        Ztd_mean = self.zd_mean(Ztd)
        Ztd_logvar = self.zd_logvar(Ztd)

        Ztd_s = self.sampling(Ztd_mean,Ztd_logvar,flag)
      
        return Ztd_s,Ztd_mean,Ztd_logvar,Ot_E_batch
 
    def emission(self,Ztd,At,Ot,Ot_E_batch):
        ### embedding (seq embedding) 
        ### check 原始 embedding size in decoder
        Yt =  self.sigmoid(self.drop_out(self.MLPs(Ztd)))
        new_zto = torch.cat((Ztd.unsqueeze(1),Ot_E_batch[:, 1:].clone()),1)
        Ot_mask =  Ot['attention_mask']
        decoder_input_ids = torch.cat((torch.tensor([self.bart.config.decoder_start_token_id],dtype = torch.long).to(Ot['input_ids'].device).unsqueeze(0), Ot['input_ids'][:,1:]),-1)

        decoder_attention_mask = Ot_mask

        Fused_Ot = self.drop_out(self.decoder(new_zto, attention_mask = Ot_mask, decoder_input_ids = decoder_input_ids,decoder_attention_mask = decoder_attention_mask))
        return Fused_Ot,decoder_input_ids,Yt

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
        # print(Ztd_last.shape)
        Ztd_list = torch.cat(Ztd_list,0).to(Ztd_last.device)

        Ztd,Ztd_mean_post,Ztd_logvar_post,Ot_E_batch = self.approximation(Ztd_list,At, Ot,label_embedding,flag)
        Ot_,Ot,Yt = self.emission(Ztd,At,Ot,Ot_E_batch)
        Ztd_mean_priori,Ztd_logvar_priori = self.trasition(Ztd_last,Ct,Ts,Add_additions)
        return Ztd,Ztd_mean_post,Ztd_logvar_post,Ot_,Ot,Yt,Ztd_mean_priori,Ztd_logvar_priori
        # return  Ot_,Yt,Ot
           

   

   