from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_gru import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel, BartPretrainedModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
# from net_text_y_priori import mllt
from base_model_rnn import mllt
from load_label_descript import label_descript
from transformers import AutoTokenizer

import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,2"


tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
self_att = "cross_att"
class_3 = True

num_epochs = 2000
max_length = 300
BATCH_SIZE = 1
class_3 = True
Test_batch_size = 1
pretrained = True
Freeze = False
SV_WEIGHTS = True
Add_additions = False
Logging = True
evaluation = True
if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
loss_ratio = [1,0,0]
Best_Roc = 0.7
Best_F1 = 0.6
visit = 'twice'
save_dir= "weights"
save_name = f"mllt_base_gru_class3_{self_att}_{visit}_0205"
if Logging:
    
    logging_text = open(f"logs/{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)
start_epoch = 0

# weight_dir = "weights/mllt_base_gru_self_att_twice_0109_epoch_12_loss_0.2352_f1_0.6054_acc_0.7485.pth"
# weight_dir = "weights/mllt_base_gru_cross_att_twice_0109_epoch_5_loss_0.4004_f1_0.6133_acc_0.7505.pth"
# weight_dir = "weights/mllt_base_gru_no_att_twice_0109_epoch_13_loss_0.318_f1_0.6058_acc_0.745.pth"
weight_dir = "weights/basemodel_clinicalbert_cluster_pretrained_class3_cross_att_once_0126_epoch_7_loss_0.4356_f1_0.8313_acc_0.7718.pth"
# weight_dir = "weights/basemodel_clinicalbert_cluster_pretrained_class3_self_att_once_0204_epoch_7_loss_0.3955_f1_0.8598_acc_0.7703.pth"

weight_dir = "weights/mllt_base_gru_class3_cross_att_twice_0205_epoch_3_loss_0.3927_f1_0.8866_acc_0.7527.pth"

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


# def collate_fn(data):
    
#     cheif_complaint_list = [d[0] for d in data]
#     text_list = [d[1][0] for d in data]
#     label_list = [d[2] for d in data]
#     return cheif_complaint_list,text_list,label_list

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

def fit(epoch,model,label_token,text_recon_loss,y_bce_loss,data_length,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc

    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)

    # if flag == 'train' and epoch ==0:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1", keep_batchnorm_fp32=True) # 这里是“欧一”，不是“零一”

    batch_loss_list = []
    batch_reconL_list = []
    chief_comp_last = deque(maxlen=2)

    batch_KLL_list = []
    batch_cls_list = []
    y_list = []
    pred_list_f1 = []
    l = 0
    one = torch.zeros(1).to(device)

    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        visit_tensor = []
        label = torch.tensor(np.array(label_list)).to(torch.float32).to(device)

        if flag == "train":
            with torch.set_grad_enabled(True):
                label_ids =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)
                label_embedding = model.encoder(**label_ids).last_hidden_state.sum(1)

                for d in range(len(text_list)):

                    text = text_list[d]
                    cheif_complaint = cheif_complaint_list[d]

                    text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    chief_comp_last.append(cheif_complaint)
                    if text['input_ids'].shape[1] > max_length:
                        text = clip_text(BATCH_SIZE,max_length,text,device)
                    elif text['input_ids'].shape[1] < max_length:
                        text = padding_text(BATCH_SIZE,max_length,text,device)
             
                    Ztd= model(label_embedding,text,self_att)
                    visit_tensor.append(Ztd)
                visit_tensor = torch.cat(visit_tensor,0).unsqueeze(0)
                all_hidden = model.transition_rnn(visit_tensor).squeeze(0)
                pred = model.emission(all_hidden)
                loss = y_bce_loss(pred.squeeze(),label.squeeze())
                y = np.array(label.cpu().data.tolist())
                pred = np.array(pred.cpu().data.tolist())
                pred=(pred > 0.5) 
                
                y_list.append(y)
                pred_list_f1.append(pred)
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss_list.append( loss.cpu().data )  
        else:
            with torch.no_grad():
                label_ids =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)
                label_embedding = model.encoder(**label_ids).last_hidden_state.sum(1)

                for d in range(len(text_list)):

                    text = text_list[d]
                    cheif_complaint = cheif_complaint_list[d]
                    text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    chief_comp_last.append(cheif_complaint)
                    if text['input_ids'].shape[1] > max_length:
                        text = clip_text(BATCH_SIZE,max_length,text,device)
                    elif text['input_ids'].shape[1] < max_length:
                        text = padding_text(BATCH_SIZE,max_length,text,device)
        
                    Ztd= model(label_embedding,text,self_att)
                    visit_tensor.append(Ztd)
                visit_tensor = torch.cat(visit_tensor,0).unsqueeze(0)
                all_hidden = model.transition_rnn(visit_tensor).squeeze(0)
                pred = model.emission(all_hidden)
                loss = y_bce_loss(pred.squeeze(),label.squeeze())
                y = np.array(label.cpu().data.tolist())
                pred = np.array(pred.cpu().data.tolist())
                pred=(pred > 0.5) 
                
                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append( loss.cpu().data )  

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
    if Logging:
        logging_text.write('%s\n'%("PHASE：{} EPOCH : {} | Micro F1 : {} | Micro ROC ： {} | Total LOSS  : {} ".format(flag,epoch + 1, f1_micro,roc_micro, total_loss)))

    print("PHASE：{} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro, total_loss))
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
    return model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro
 


if __name__ == '__main__':
    train_dataset = PatientDataset(f"dataset/new_packed_data/{visit}/", class_3 = class_3, visit = visit, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f"dataset/new_packed_data/{visit}/",class_3 = class_3, visit = visit, flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    label_embedding = label_descript()

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

            if i == 10:
                for param in child.parameters():
                    param.requires_grad = False
    ##########################



    text_recon_loss = nn.CrossEntropyLoss()
    y_bce_loss = nn.BCELoss()
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
    
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro  = fit(epoch,model,label_embedding,text_recon_loss,y_bce_loss,test_dataset.__len__(),testloader,optimizer,flag='test')
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

            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,label_embedding,text_recon_loss,y_bce_loss,train_dataset.__len__(),trainloader,optimizer,flag='train')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,label_embedding,text_recon_loss,y_bce_loss,test_dataset.__len__(),testloader,optimizer,flag='test')


        







