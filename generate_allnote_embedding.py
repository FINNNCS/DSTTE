import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_cluster_new import PatientDataset

import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel, BartPretrainedModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
# from net_text_y_priori import mllt
from embed_generator import mllt
# from clinical_bert import mllt

from transformers import AutoTokenizer
from load_label_descript import label_descript
import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="1"
from sklearn.cluster import KMeans

from transformers import AutoTokenizer, AutoModel

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)
self_att = "cross_att"
num_epochs = 1
max_length = 300
BATCH_SIZE = 300
evaluation = False
pretrained = True
Freeze = False
SV_WEIGHTS = False
Logging = False
if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
loss_ratio = [1,0,0]
Best_Roc = 0.7
Best_F1 = 0.6
visit = 'twice'
save_dir= "weights"
save_name = f"clinical_bert_basemodel_{self_att}_{visit}_0117"
if Logging:
    logging_text = open(f"{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)

start_epoch = 0

weight_dir = "weights/basemodel_cluster_pretrained_class3_cross_att_once_0123_epoch_18_loss_0.5083_f1_0.8511_acc_0.7289.pth"


def get_kmeans_centers(all_embeddings, num_classes):
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    return clustering_model.cluster_centers_

def clip_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    seq_ids = input_ids[:,[-1]]
    seq_mask = attention_mask[:,[-1]]
    input_ids_cliped = input_ids[:,:max_length-1]
    attention_mask_cliped = attention_mask[:,:max_length-1]
    input_ids_cliped = torch.cat([input_ids_cliped,seq_ids],dim=-1)
    attention_mask_cliped = torch.cat([attention_mask_cliped,seq_mask],dim=-1)
    vec = {'input_ids': input_ids_cliped,
    'attention_mask': attention_mask_cliped}
    return vec

def padding_text(batch_size,max_length,vec,device):
    input_ids = vec['input_ids']
    attention_mask = vec['attention_mask']
    sentence_difference = max_length - input_ids.shape[1]
    padding_ids = torch.ones((batch_size,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((batch_size,sentence_difference), dtype = torch.long).to(device)
    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)

    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec

def padding_event(vec):
    max_word_length = max([e['input_ids'].shape[0] for e in vec])
    max_subword_length = max([e['input_ids'].shape[1] for e in vec])
    tmp = []
    for e in vec:
        input_ids = e['input_ids']
        attention_mask = e['attention_mask']
        word_difference = max_word_length - input_ids.shape[0]
        sub_word_difference = max_subword_length - input_ids.shape[1]
        sub_padding_ids = torch.ones((input_ids.shape[0],sub_word_difference), dtype = torch.long ).to(f"cuda:{input_ids.get_device()}")
        sub_padding_mask = torch.zeros((input_ids.shape[0],sub_word_difference), dtype = torch.long).to(f"cuda:{input_ids.get_device()}")
        input_ids_sub_padded = torch.cat([input_ids,sub_padding_ids],dim=-1)
        attention_mask_sub_padded = torch.cat([attention_mask,sub_padding_mask],dim=-1)
        word_padding_ids = torch.cat(( torch.zeros((word_difference,1), dtype = torch.long).to(f"cuda:{input_ids.get_device()}"),torch.ones((word_difference,max_subword_length-1), dtype = torch.long ).to(f"cuda:{input_ids.get_device()}")),dim=-1)
        word_padding_mask = torch.zeros((word_difference,max_subword_length), dtype = torch.long).to(f"cuda:{input_ids.get_device()}")
        input_ids_padded = torch.cat([input_ids_sub_padded,word_padding_ids],dim=0)
        attention_mask_padded = torch.cat([attention_mask_sub_padded,word_padding_mask],dim=0)
        vec = {'input_ids': input_ids_padded,
        'attention_mask': attention_mask_padded}
        tmp.append(vec)
    return tmp



def collate_fn(data):
    
    cheif_complaint_list = [d[0] for d in data]
    text_list = [d[1][0] for d in data]
    label_list = [d[2] for d in data]
    return cheif_complaint_list,text_list,label_list


def kl_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post,one):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1))

        # KLD = torch.sum(0.5*(Z_logvar_post-Z_logvar_prioir+(torch.exp(Z_logvar_prioir)+(Z_mean_post-Z_mean_prioir).pow(2))/torch.exp(Z_logvar_post)-one), 1)  
        # torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)  
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL



def fit(epoch,model,label_embedding,regularization_loss,y_bce_loss,data_length,dataloader,optimizer,flag='train'):
    device = device1
    model.eval()
    model.to(device)
    embed_list = []
    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
    

        with torch.no_grad():

            text = tokenizer(text_list, return_tensors="pt",padding=True,max_length = max_length).to(device)
            label_token =  tokenizer(label_embedding, return_tensors="pt",padding=True,max_length = max_length).to(device)

            if text['input_ids'].shape[1] > max_length:
                text = clip_text(BATCH_SIZE,max_length,text,device)
            elif text['input_ids'].shape[1] < max_length:
                text = padding_text(BATCH_SIZE,max_length,text,device)


            embed = model(text,label_token)
            embed_list.append(embed)
    
    embed_list = torch.cat(embed_list,0).cpu()
    torch.save(embed_list, "dataset/medical_note_embedding_train_twice.pth")

    # cluster_centers = torch.tensor(get_kmeans_centers(embed_list,8))
    # print("after: ",cluster_centers[:2,:10])

    # torch.save(cluster_centers, "dataset/medical_note_embedding_kmeans_8_mean.pth")

if __name__ == '__main__':
    eval_visit = "four"
    label_embedding = label_descript()

    train_dataset = PatientDataset(f"dataset/new_packed_data/{visit}/", class_3 = False, visit = visit, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = False)

    train_length = train_dataset.__len__()

    print(train_length)

    model = mllt()

    if pretrained:
        print(f"loading weights: {weight_dir}")
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device1)), strict=False)

    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”

    ### freeze parameters ####
    optimizer = optim.Adam(model.parameters(True), lr = 1e-5)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            if i == 15:
                for param in child.parameters():
                    param.requires_grad = False
    ##########################



    text_recon_loss = nn.CrossEntropyLoss()
    y_bce_loss = nn.BCELoss()
    regularization_loss = nn.CrossEntropyLoss()

    fit(1,model,label_embedding,regularization_loss,y_bce_loss,train_length,trainloader,optimizer,flag='train')



   

 







