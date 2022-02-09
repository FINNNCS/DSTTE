from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_transition import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel, BartPretrainedModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from transition_cluster_model import mllt
from load_label_descript import label_descript
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,3"
import matplotlib.pyplot as plt


tsne = TSNE(random_state=0, perplexity=10)
palette = sns.color_palette("bright",25)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)

num_epochs = 2000
max_length = 300
Add_additions = False
class_3 = True
cluster = True
self_att = False
BATCH_SIZE = 1
# BATCH_SIZE = 60
cluster_number = 8
Freeze = True
evaluation = True
pretrained = True
SV_WEIGHTS = False
Logging = False
if self_att:
    from transition_selfatt import mllt

# loss_ratio = [0.9,0.1,0]
# loss_ratio = [0.999,0.001,1]
loss_ratio = [1,1e-4,1]

Best_Roc = 0.7
Best_F1 = 0.6
visit = 'twice'
save_dir= "weights"
save_name = f"mllt_clinical_bert_pretrain__load_cluster_weights_transition_cluster_batch_{str(BATCH_SIZE)}_label_att_batch_{visit}_0129"
logging_text = open(f"logs/{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)
start_epoch = 0

# weight_dir = "weights/mllt_clinical_bert_transition_label_att_twice_0126_epoch_3_loss_0.1907_f1_0.8587_acc_0.6808.pth"
# weight_dir = "weights/mllt_clinical_bert_pretrainer_class3_fusedweighted.pth"

weight_dir = "weights/mllt_clinical_bert_pretrain_transition_cluster_batch_30_label_att_batch_twice_0128_epoch_18_loss_0.5444_f1_0.891_acc_0.6515.pth"

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
    sentence_difference = max_length - len(input_ids[0])
    padding_ids = torch.ones((1,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((1,sentence_difference), dtype = torch.long).to(device)

    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)
    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec

def collate_fn(data):
    # cheif_complaint_list = [d[0] for d in data]
    # text_list = [d[1] for d in data]
    # label_list =[d[2] for d in data]
    cheif_complaint_list = [data[0][0]]
    text_list = [data[0][1]]
    label_list =[data[0][2]]
    return cheif_complaint_list,text_list,label_list


def KL_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1)).to(f"cuda:{Z_mean_prioir.get_device()}")
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL


def fit(epoch,dataset,model,center_embedding,label_token,text_recon_loss,y_bce_loss,cluster_loss,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc

    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)
    cluster_loss.to(device)

    # if flag == 'train' and epoch ==0:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1", keep_batchnorm_fp32=True) # 这里是“欧一”，不是“零一”

    chief_comp_last = deque(maxlen=2)

    y_list = []
    pred_list_f1 = []
    l = 0
    eopch_loss_list = []
    epoch_classify_loss_list = []
    epoch_kl_loss_list = []
    epoch_cluster_loss_list = []
    all_embedding = []
    all_embedding = torch.load("results/all_test_medical_embedding.pth").data.numpy()
    Center = torch.load("results/cluster_center.pth",map_location=torch.device(device2)).cpu().data.numpy()
    
    # Center = Center.cpu().data.numpy()
    X_tsne = tsne.fit_transform(all_embedding)

    # X_tsne3 = scaler.fit_transform(X_tsne3.squeeze())
    # y = [i for i in range(1,8+1)]
    # sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(),c=y)
    # sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze())
    visit_label = []
    for d in range(all_embedding.shape[0]):
        single_visit = all_embedding[d,:]
        cluster_id = [np.sqrt(np.sum(np.square(single_visit - Center[c,:])))  for c in range(8)]
        visit_label.append(cluster_id.index( min(cluster_id)))

    # for t in range(len(X_tsne)):
        # plt.text(X_tsne[t:t+1,0].squeeze(),X_tsne[t:t+1,1].squeeze()+0.05,size = "large")

        # plt.text(X_tsne[t:t+1,0].squeeze(),X_tsne[t:t+1,1].squeeze()+0.05,y[t],size = "large")
    sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(),c=visit_label)
    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
        if i == 1: break
        optimizer.zero_grad()
        center_embedding = torch.nn.Parameter(center_embedding).to(device)
        batch_KLL_list = torch.zeros(len(text_list)).to(device)
        batch_cls_list = torch.zeros(len(text_list)).to(device)
        phi_list = []
        patient_z = []
        with torch.no_grad():
            # print(len(text_list))
            for p in range(len(text_list)):
                p_text = text_list[p]
                p_label = label_list[p]
                Ztd_zero = torch.randn((1, model.hidden_size//2)).to(device)
                Ztd_zero.requires_grad = True
                Kl_loss = torch.zeros(len(p_text)).to(device)
                cls_loss = torch.zeros(len(p_text)).to(device)

                Ztd_last = Ztd_zero
                Ztd_list = [Ztd_zero]
                label_ids =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)

                label_embedding = model.encoder(**label_ids).last_hidden_state.sum(1)
            
                for v in range(len(p_text)):
                
                    text = p_text[v]
                    label = p_label[v]
                    # cheif_complaint = cheif_complaint_list[d]
                    label = torch.tensor(label).to(torch.float32).to(device)
                    text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    # cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    # chief_comp_last.append(cheif_complaint)
                    # print(text)
                    if text['input_ids'].shape[1] > max_length:
                        text = clip_text(BATCH_SIZE,max_length,text,device)
                    elif text['input_ids'].shape[1] < max_length:
                        text = padding_text(BATCH_SIZE,max_length,text,device)
                    if v == 0:
                        Ztd_last = Ztd_zero
                    phi,Center,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                    model(center_embedding,Ztd_list,label_embedding,chief_comp_last,text,Ztd_last,flag,cluster)
                    patient_z.append(Ztd.squeeze().cpu().data.numpy())

                    # phi_list.append(phi)
                    # all_embedding.append(Ztd.squeeze().cpu().data.numpy())
                  
            patient_z = np.stack(patient_z)
            # for z in all_embedding:
            #     single_z = 
            # patient_z_tsne = tsne.fit_transform(patient_z)

            # for t in range(len(patient_z)):
                # plt.text(X_tsne[t:t+1,0].squeeze(),X_tsne[t:t+1,1].squeeze()+0.05,[y[t]],size = "large")
            # sns.scatterplot(patient_z_tsne[:,0].squeeze(), patient_z_tsne[:,1].squeeze())

    plt.savefig(f'images/tsne_cluster.jpg')
    # print(dataset.)



if __name__ == '__main__':
    # train_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="train")
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    # test_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="test")
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    # test_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/new_packed_data/{visit}/',class_3,visit,flag="test")

    test_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/cluster_eval_data/{visit}/',class_3,visit,flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    label_embedding = label_descript()
    center_embedding = torch.load("dataset/medical_note_embedding_kmeans_8.pth").type(torch.cuda.FloatTensor)

    print(test_dataset.__len__())

    model = mllt(class_3)

    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
        print("loading weight: ",weight_dir)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”

    ### freeze parameters ####
    optimizer = optim.Adam(model.parameters(True), lr = 1e-5)

    if Freeze:
        for (i,child) in enumerate(model.children()):
            # print(i,child)
            if i == 10:
                # print(child)
                for param in child.parameters():
                    param.requires_grad = False
    ##########################



    text_recon_loss = nn.CrossEntropyLoss()
    y_bce_loss = nn.BCELoss()
    cluster_loss = nn.KLDivLoss(reduction='sum')

    precision_micro_list = []
    precision_macro_list = []
    recall_micro_list = []
    recall_macro_list = []
    f1_micro_list = []
    f1_macro_list = []
    roc_micro_list = []
    roc_macro_list = []

    fit(1,test_dataset,model,center_embedding,label_embedding,text_recon_loss,y_bce_loss,cluster_loss,testloader,optimizer,flag='test')
    
    







