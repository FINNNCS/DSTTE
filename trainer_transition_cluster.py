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

import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,2"


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)

num_epochs = 2000
max_length = 300
Add_additions = False
class_3 = True
cluster = False
self_att = False
BATCH_SIZE = 1
Test_batch_size = 1
pretrained = True
Freeze = False
SV_WEIGHTS = True
Logging = False
evaluation = False
if evaluation:
    pretrained = False
    SV_WEIGHTS = False
    Logging = False
attention = "self_att"

# loss_ratio = [0.9,0.1,0]
loss_ratio = [0.999,0.001,1]

Best_Roc = 0.7
Best_F1 = 0.6
visit = 'twice'
save_dir= "weights"
save_name = f"mllt_clinical_bert_pretrain_transition_cluster_{attention}_{visit}_0126"
logging_text = open(f"logs/{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)
start_epoch = 0
# weight_dir = "weights/mllt_transition_twice_twice_1222_epoch_885_loss_2.5294_f1_0.6093_acc_0.7512.pth"
# weight_dir = "weights/basemodel_clinicalbert_cluster_pretrained_class3_cross_att_once_0126_epoch_7_loss_0.4356_f1_0.8313_acc_0.7718.pth"

# weight_dir = "weights/mllt_transition_label_embedding_twice_0102_epoch_42_loss_0.4577_f1_0.6091_acc_0.7531.pth"
# weight_dir = "weights/mllt_transition_label_embedding_twice_0102_epoch_0_loss_0.4336_f1_0.5999_acc_0.7446.pth"
# weight_dir = "weights/mllt_base_label_attention_once_0102_epoch_7_loss_0.3449_f1_0.5928_acc_0.7522.pth"
# 
# weight_dir = "weights/mllt_transition_label_embedding_twice_0102_epoch_42_loss_0.4577_f1_0.6091_acc_0.7531.pth"
# weight_dir = "weights/mllt_transition_self_att_twice_0106_epoch_20_loss_0.3398_f1_0.6061_acc_0.7477.pth"
# weight_dir = "weights/mllt_clinical_bert_transition_label_att_twice_0126_epoch_3_loss_0.1907_f1_0.8587_acc_0.6808.pth"
# weight_dir = "weights/basemodel_clinicalbert_cluster_pretrained_class3_cross_att_once_0126_epoch_7_loss_0.4356_f1_0.8313_acc_0.7718.pth"
weight_dir = "weights/basemodel_clinicalbert_cluster_pretrained_class3_self_att_once_0204_epoch_4_loss_0.354_f1_0.8741_acc_0.7511.pth"

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
    padding_ids = torch.ones((batch_size,sentence_difference), dtype = torch.long ).to(device)
    padding_mask = torch.zeros((batch_size,sentence_difference), dtype = torch.long).to(device)

    input_ids_padded = torch.cat([input_ids,padding_ids],dim=-1)
    attention_mask_padded = torch.cat([attention_mask,padding_mask],dim=-1)
    vec = {'input_ids': input_ids_padded,
    'attention_mask': attention_mask_padded}
    return vec


def collate_fn(data):
    
    cheif_complaint_list = data[0][0]
    text_list = data[0][1]
    label_list = data[0][2]

    return cheif_complaint_list,text_list,label_list



def KL_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post,one):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1)).to(f"cuda:{Z_mean_prioir.get_device()}")
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL

def fit(epoch,model,center_embedding,label_token,text_recon_loss,y_bce_loss,cluster_loss,dataloader,optimizer,flag='train'):
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
    batch_loss_list = []
    batch_cluster_list = []
    batch_KLL_list = []
    batch_cls_list = []
    y_list = []
    pred_list_f1 = []
    l = 0
    one = torch.zeros(1).to(device)

    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
     
        Ztd_zero = torch.randn((1, model.hidden_size//2)).to(device)
        Ztd_zero.requires_grad = True

        center_embedding = torch.nn.Parameter(center_embedding).to(device)

        Kl_loss = torch.zeros(len(text_list)).to(device)
        cls_loss = torch.zeros(len(text_list)).to(device)
        clus_loss = torch.zeros(len(text_list)).to(device)

        Ztd_last = Ztd_zero
        # label_embedding = model.encoder(**label_token.to(device)).last_hidden_state.sum(1)
        # label_embedding = label_embedding.detach()
        Ztd_list = [Ztd_zero]
        label_ids =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)

        label_embedding = model.encoder(**label_ids).last_hidden_state.sum(1)

        if flag == "train":
            with torch.set_grad_enabled(True):
                for d in range(len(text_list)):

                 
                    text = text_list[d]
                    label = label_list[d]
                    cheif_complaint = cheif_complaint_list[d]

                    label = torch.tensor(label).to(torch.float32).to(device)
                    text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    chief_comp_last.append(cheif_complaint)
                    if text['input_ids'].shape[1] > max_length:
                        text = clip_text(BATCH_SIZE,max_length,text,device)
                    elif text['input_ids'].shape[1] < max_length:
                        text = padding_text(BATCH_SIZE,max_length,text,device)

             
                    # event_list = tokenizer(event_codes, return_tensors="pt",padding=True).to(device)
                    # print(event_list['input_ids'].shape)
    
                    if d == 0:
                        Ztd_last = Ztd_zero
                    if cluster:
                        phi,cluster_target,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,label_embedding,chief_comp_last,text,Ztd_last,flag,cluster,attention)
                    else:
                        Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,label_embedding,chief_comp_last,text,Ztd_last,flag,cluster,attention)
                    Ztd_last = Ztd_mean_post
                    Ztd_list.append(Ztd_last)

                    icd_L = y_bce_loss(pred.squeeze(),label.squeeze())
                    if d == 0:
                        q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                    else:
                        q_ztd = KL_loss(Ztd_logvar_priori,Ztd_mean_priori,Ztd_mean_post, Ztd_logvar_post,one)
                    if cluster:

                        cluster_lss = cluster_loss((phi+1e-08).log(),cluster_target)/phi.shape[0]
                    Kl_loss[d] = q_ztd
                    cls_loss[d] = icd_L
                    if cluster:
                        clus_loss[d] = cluster_lss
                    label = np.array(label.cpu().data.tolist())
                    pred = np.array(pred.cpu().data.tolist())
                    pred=(pred > 0.5) 
                    y_list.append(label)
                    pred_list_f1.append(pred)


                cls_loss_p = cls_loss.view(-1).mean()
                kl_loss_p = Kl_loss.view(-1).mean()
                clus_loss_p = clus_loss.view(-1).mean()
                batch_cls_list.append(cls_loss_p.cpu().data )
                batch_KLL_list.append(kl_loss_p.cpu().data )
                batch_cluster_list.append(clus_loss_p.cpu().data )
                if cluster:
                    total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p + loss_ratio[2]*clus_loss_p
                else:
                    total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p
                # with amp.scale_loss(total_loss, optimizer) as total_loss:

                ###### https://github.com/guxd/deepHMM/blob/master/models/dhmm.py
                total_loss.backward(retain_graph=True)
                optimizer.step()
                loss = total_loss.cpu().data 
                batch_loss_list.append(loss)   
                l += 1
        else:
            with torch.no_grad():
                for d in range(len(text_list)):
                    text = text_list[d]
                    label = label_list[d]
                    cheif_complaint = cheif_complaint_list[d]
                    label = torch.tensor(label).to(torch.float32).to(device)
                    text = tokenizer(text, return_tensors="pt",padding=True, max_length=max_length).to(device)
                    cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    chief_comp_last.append(cheif_complaint)
                    # label_embedding =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)

                    if text['input_ids'].shape[1] > max_length:
                        text = clip_text(BATCH_SIZE,max_length,text,device)
                    elif text['input_ids'].shape[1] < max_length:
                        text = padding_text(BATCH_SIZE,max_length,text,device)
                    # event_list = []
                    # for e in event_codes:
                    #     e = tokenizer(e, return_tensors="pt",padding=True).to(device)
                    #     event_list.append(e)
                    # event_codes = event_codes_list[d]


                    if d == 0:
                        Ztd_last = Ztd_zero
                    if cluster:
                        phi,cluster_target,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,label_embedding,chief_comp_last,text,Ztd_last,flag,cluster,attention)
                    else:
                        Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
                        model(center_embedding,Ztd_list,label_embedding,chief_comp_last,text,Ztd_last,flag,cluster,attention)
                    Ztd_last = Ztd_mean_post
                    Ztd_list.append(Ztd_last)

                    icd_L = y_bce_loss(pred.squeeze(),label.squeeze())
                    if d == 0:
                        q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                    else:
                        q_ztd = KL_loss(Ztd_logvar_priori,Ztd_mean_priori,Ztd_mean_post, Ztd_logvar_post,one)
                    if cluster:
                        cluster_lss = cluster_loss((phi+1e-08).log(),cluster_target)/phi.shape[0]
                        # print(phi[:1,:],cluster_target[:,])
                        # print(cluster_lss)
                    Kl_loss[d] = q_ztd
                    cls_loss[d] = icd_L
                    if cluster:
                        clus_loss[d] = cluster_lss
                    label = np.array(label.cpu().data.tolist())
                    pred = np.array(pred.cpu().data.tolist())
                    pred=(pred > 0.5) 
                    y_list.append(label)
                    pred_list_f1.append(pred)


                cls_loss_p = cls_loss.view(-1).mean()
                kl_loss_p = Kl_loss.view(-1).mean()
                clus_loss_p = clus_loss.view(-1).mean()
                batch_cls_list.append(cls_loss_p.cpu().data )
                batch_KLL_list.append(kl_loss_p.cpu().data )
                batch_cluster_list.append(clus_loss_p.cpu().data )
                if cluster:

                    total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p + loss_ratio[2]*clus_loss_p
                else:
                    total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p

                loss = total_loss.cpu().data 
                batch_loss_list.append(loss)   
                l+=1
    y_list = np.vstack(y_list)
    pred_list_f1 = np.vstack(pred_list_f1)    

    precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
    recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
    precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
    recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_f1,average="micro")
    f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(y_list,pred_list_f1,average="macro")
    total_loss = sum(batch_loss_list) / len(batch_loss_list)
    total_cls = sum(batch_cls_list) / len(batch_cls_list)
    total_kls = sum(batch_KLL_list) / len(batch_KLL_list)
    total_clus = sum(batch_cluster_list) / len(batch_cluster_list)

    if Logging:
            logging_text.write('%s\n'%("PHASE：{} EPOCH : {} | | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | Total LOSS  : {} | Total Cls LOSS  : {} | Total KL LOSS  : {} ".format(flag,epoch + 1,  f1_micro,f1_macro,roc_micro,roc_macro, total_loss,total_cls,total_kls)))
    print("PHASE：{} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | Total LOSS  : {} | Total Cls LOSS  : {} | Total KL LOSS  : {} | Total Cluster LOSS  : {} ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro, total_loss,total_cls,total_kls,total_clus))
    if flag == 'test':
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"/home/comp/cssniu/mllt/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
            elif roc_micro > Best_Roc:
                Best_Roc = roc_micro
                PATH=f"/home/comp/cssniu/mllt/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
    return  model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro    


if __name__ == '__main__':
    # train_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="train")
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    # test_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="test")
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    train_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/new_packed_data/{visit}/',class_3,visit,flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    test_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/new_packed_data/{visit}/',class_3,visit,flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    label_embedding = label_descript()
    center_embedding = torch.load("dataset/medical_note_embedding_kmeans_8.pth").type(torch.cuda.FloatTensor)

    print(train_dataset.__len__())
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
            print(child,i)

    #         if i == 10:
    #             print(child)
    #             for param in child.parameters():
    #                 param.requires_grad = False
    # ##########################



    text_recon_loss = nn.CrossEntropyLoss()
    y_bce_loss = nn.BCELoss()
    cluster_loss = nn.KLDivLoss(reduction='sum')

    if evaluation:
        precision_micro_list = []
        precision_macro_list = []
        recall_micro_list = []
        recall_macro_list = []
        f1_micro_list = []
        f1_macro_list = []
        roc_micro_list = []
        roc_macro_list = []
        for epoch in range(5):
    
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro  = fit(epoch,model,center_embedding,label_embedding,text_recon_loss,y_bce_loss,cluster_loss,testloader,optimizer,flag='test')
            precision_micro_list.append(precision_micro)
            precision_macro_list.append(precision_macro)
            recall_micro_list.append(recall_micro)
            recall_macro_list.append(recall_macro)            
            f1_micro_list.append(f1_micro)
            f1_macro_list.append(f1_macro)                    
            roc_micro_list.append(roc_micro)
            roc_macro_list.append(roc_macro)
        precision_micro_mean = np.mean(precision_micro_list)
        precision_macro_mean = np.mean(precision_macro_list)        
        recall_micro_mean = np.mean(recall_micro_list)
        recall_macro_mean = np.mean(recall_macro_list)        
        f1_micro_mean = np.mean(f1_micro_list)
        f1_macro_mean = np.mean(f1_macro_list)
        roc_micro_mean = np.mean(roc_micro_list)
        roc_macro_mean = np.mean(roc_macro_list)
        print(" Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {}  ".format(precision_micro_mean,precision_macro_mean,recall_micro_mean,recall_macro_mean,f1_micro_mean,f1_macro_mean,roc_micro_mean,roc_macro_mean))

    else:
        for epoch in range(start_epoch,num_epochs):

            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,label_embedding,text_recon_loss,y_bce_loss,cluster_loss,trainloader,optimizer,flag='train')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,center_embedding,label_embedding,text_recon_loss,y_bce_loss,cluster_loss,testloader,optimizer,flag='test')



    







