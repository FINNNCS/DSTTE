import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_new import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel, BartPretrainedModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
# from net_text_y_priori import mllt
from trasition_recon_model import mllt

import copy
# from apex import amp

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="2,3"



tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)

num_epochs = 2000
max_length = 300

BATCH_SIZE = 1
Test_batch_size = 1
pretrained = True
Freeze = False
SV_WEIGHTS = True
Add_additions = False
Logging = True
loss_ratio = [0.98,0.01,0.01]
Best_Roc = 0.7
Best_F1 = 0.6
visit = 'twice'
save_dir= "weights"
save_name = f"mllt_transition_recon_pretrained_{visit}_1229"
logging_text = open(f"{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:1" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
start_epoch = 0
weight_dir = "weights/mllt_transition_twice_twice_1222_epoch_885_loss_2.5294_f1_0.6093_acc_0.7512.pth"



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
    
    cheif_complaint_list = data[0][0]
    text_list = data[0][1]
    label_list = data[0][2]
    event_codes =  data[0][3]
    time_stamp_list = data[0][4]
    return cheif_complaint_list,text_list,label_list,event_codes,time_stamp_list



def kl_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post,one):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1)).to(f"cuda:{Z_mean_prioir.get_device()}")

        # KLD = torch.sum(0.5*(Z_logvar_post-Z_logvar_prioir+(torch.exp(Z_logvar_prioir)+(Z_mean_post-Z_mean_prioir).pow(2))/torch.exp(Z_logvar_post)-one), 1)  
        # torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)  
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL

def fit(epoch,model,text_recon_loss,y_bce_loss,data_list,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc

    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    y_bce_loss.to(device)
    text_recon_loss.to(device)
    # if flag == 'train' and epoch ==0:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1", keep_batchnorm_fp32=True) # 这里是“欧一”，不是“零一”
    chief_comp_last = deque(maxlen=2)

    batch_loss_list = []
    batch_reconL_list = []
    batch_KLL_list = []
    batch_cls_list = []

    y_list = []
    pred_list_f1 = []
    l = 0
    one = torch.zeros(1).to(device)

    for i,(cheif_complaint_list,text_list,label_list,event_codes_list,time_stamp_list) in enumerate(tqdm(dataloader)):

        optimizer.zero_grad()
     
        Ztd_zero = torch.randn((1, model.hidden_size)).to(device)
        Ztd_zero.requires_grad = True
        Zto_zero = torch.randn((1, model.hidden_size)).to(device)
        Zto_zero.requires_grad = True

        Kl_loss = torch.zeros(len(text_list)).to(device)
        cls_loss = torch.zeros(len(text_list)).to(device)
        RC_loss = torch.zeros(len(text_list)).to(device)

        Ztd_last,Zto_last = Ztd_zero,Zto_zero
        # y_list = np.zeros((len(text_list),15))
        # pred_list = np.zeros((len(text_list),15))
        if flag == "train":
            with torch.set_grad_enabled(True):
                for d in range(len(text_list)):
                    text = text_list[d]
                    label = label_list[d]
                    event_codes = event_codes_list[d]
                    cheif_complaint = cheif_complaint_list[d]
                    time_stamp = torch.tensor(time_stamp_list[d]).to(torch.float32).to(device)

                    label = torch.tensor(label).to(torch.float32).to(device)
                    text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    chief_comp_last.append(cheif_complaint)
                    if text['input_ids'].shape[1] > max_length:
                        text = clip_text(BATCH_SIZE,max_length,text,device)
                    elif text['input_ids'].shape[1] < max_length:
                        text = padding_text(BATCH_SIZE,max_length,text,device)

             
                    event_list = tokenizer(event_codes, return_tensors="pt",padding=True).to(device)
                    # print(event_list['input_ids'].shape)
    
                    if d == 0:
                        Ztd_last,Zto_last = Ztd_zero,Zto_zero

                    Ot_,Ot,Zto,Ztd,Zto_mean_post,Zto_logvar_post,Ztd_mean_post,Ztd_logvar_post,pred,Zto_mean_priori,Zto_logvar_priori,Ztd_mean_priori,Ztd_logvar_priori = \
                    model(time_stamp,chief_comp_last,event_list,text,Zto_last,Ztd_last,Add_additions,flag)
                    Ztd_last,Zto_last = Ztd_mean_post,Zto_mean_post

                    icd_L = y_bce_loss(pred.squeeze(),label.squeeze())
                    text_reconL = reconstrution_loss(text_recon_loss,Ot_,Ot)
                    if d == 0:

                        q_zto = torch.mean(-0.5 * torch.sum(1 + Zto_logvar_post - Zto_mean_post ** 2 - Zto_logvar_post.exp(), dim = 1), dim = 0)
                        q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                    else:
                        q_zto = kl_loss(Zto_mean_priori,Zto_logvar_priori,Zto_mean_post, Zto_logvar_post,one)
                        q_ztd = kl_loss(Ztd_logvar_priori,Ztd_mean_priori,Ztd_mean_post, Ztd_logvar_post,one)

                    Kl_loss[d] = (q_zto+q_ztd)/2
                    cls_loss[d] = icd_L
                    RC_loss[d] =  text_reconL

                    y = np.array(label.cpu().data.tolist())
                    pred = np.array(pred.cpu().data.tolist())
                    pred=(pred > 0.5) 
                    y_list.append(label)
                    pred_list_f1.append(pred)

                cls_loss_p = cls_loss.view(-1).mean()
                kl_loss_p = Kl_loss.view(-1).mean()
                rec_loss_p = RC_loss.view(-1).mean()

                batch_cls_list.append(cls_loss_p.cpu().data )
                batch_KLL_list.append(kl_loss_p.cpu().data )
                batch_reconL_list.append(rec_loss_p.cpu().data )
                total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p + loss_ratio[2]*rec_loss_p
                # with amp.scale_loss(total_loss, optimizer) as total_loss:

                ###### https://github.com/guxd/deepHMM/blob/master/models/dhmm.py
                total_loss.backward(retain_graph=True)
                optimizer.step()
                loss = total_loss.cpu().data 
                batch_loss_list.append(loss)   
        else:
            with torch.no_grad():
                for d in range(len(text_list)):
                    text = text_list[d]
                    label = label_list[d]
                    event_codes = event_codes_list[d]
                    cheif_complaint = cheif_complaint_list[d]
                    time_stamp = torch.tensor(time_stamp_list[d]).to(torch.float32).to(device)
                    label = torch.tensor(label).to(torch.float32).to(device)
                    text = tokenizer(text, return_tensors="pt",padding=True, max_length=max_length).to(device)
                    cheif_complaint =  tokenizer(cheif_complaint, return_tensors="pt",padding=True,max_length = max_length).to(device)
                    chief_comp_last.append(cheif_complaint)

                    if text['input_ids'].shape[1] > max_length:
                        text = clip_text(BATCH_SIZE,max_length,text,device)
                    elif text['input_ids'].shape[1] < max_length:
                        text = padding_text(BATCH_SIZE,max_length,text,device)
                    # event_list = []
                    # for e in event_codes:
                    #     e = tokenizer(e, return_tensors="pt",padding=True).to(device)
                    #     event_list.append(e)
                    # event_codes = event_codes_list[d]

        
                    event_list = tokenizer(event_codes, return_tensors="pt",padding=True).to(device)
                    if d == 0:
                        Ztd_last,Zto_last = Ztd_zero,Zto_zero

                    Ot_,Ot,Zto,Ztd,Zto_mean_post,Zto_logvar_post,Ztd_mean_post,Ztd_logvar_post,pred,Zto_mean_priori,Zto_logvar_priori,Ztd_mean_priori,Ztd_logvar_priori = \
                    model(time_stamp,chief_comp_last,event_list,text,Zto_last,Ztd_last,Add_additions,flag)

                    Ztd_last,Zto_last = Ztd_mean_post,Zto_mean_post

                    icd_L = y_bce_loss(pred.squeeze(),label.squeeze())
                    text_reconL = reconstrution_loss(text_recon_loss,Ot_,Ot)

                    if d == 0:
                        q_zto = torch.mean(-0.5 * torch.sum(1 + Zto_logvar_post - Zto_mean_post ** 2 - Zto_logvar_post.exp(), dim = 1), dim = 0)
                        q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                    else:
                        q_zto = kl_loss(Zto_mean_priori,Zto_logvar_priori,Zto_mean_post, Zto_logvar_post,one)
                        q_ztd = kl_loss(Ztd_logvar_priori,Ztd_mean_priori,Ztd_mean_post, Ztd_logvar_post,one)
                    Kl_loss[d] = (q_zto+q_ztd)/2
                    cls_loss[d] = icd_L
                    RC_loss[d] =  text_reconL

                    label = np.array(label.cpu().data.tolist())
                    pred = np.array(pred.cpu().data.tolist())
                    pred=(pred > 0.5) 
                    y_list.append(label)
                    pred_list_f1.append(pred)


                cls_loss_p = cls_loss.view(-1).mean()
                kl_loss_p = Kl_loss.view(-1).mean()
                rec_loss_p = RC_loss.view(-1).mean()

                batch_cls_list.append(cls_loss_p.cpu().data )
                batch_KLL_list.append(kl_loss_p.cpu().data )
                batch_reconL_list.append(rec_loss_p.cpu().data )
                total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p + loss_ratio[2]*rec_loss_p
                loss = total_loss.cpu().data 
                batch_loss_list.append(loss)   
          
    y_list = np.vstack(y_list)
    pred_list_f1 = np.vstack(pred_list_f1)      
    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_f1,average="micro")
    total_loss = sum(batch_loss_list) / len(batch_loss_list)
    total_cls = sum(batch_cls_list) / len(batch_cls_list)
    total_kls = sum(batch_KLL_list) / len(batch_KLL_list)
    total_Recon = sum(batch_reconL_list) / len(batch_reconL_list)
    if Logging:
            logging_text.write('%s\n'%("PHASE：{} EPOCH : {} | Micro F1 : {} | Micro ROC ： {} | Total LOSS  : {} | Total Cls LOSS  : {} | Total KL LOSS  : {} | Total Recon LOSS  : {}".format(flag,epoch + 1, f1_micro,roc_micro, total_loss,total_cls,total_kls,total_Recon)))
    print("PHASE：{} EPOCH : {} | Micro F1 : {} | Micro ROC ： {} | Total LOSS  : {} | Total Cls LOSS  : {} | Total KL LOSS  : {} | Total Recon LOSS  : {}".format(flag,epoch + 1, f1_micro,roc_micro, total_loss,total_cls,total_kls,total_Recon))
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
    return model       


if __name__ == '__main__':
    # train_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="train")
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    # test_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="test")
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    train_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/new_packed_data/{visit}/',visit,flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    test_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/new_packed_data/{visit}/',visit,flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    print(train_dataset.__len__())
    print(test_dataset.__len__())

    model = mllt(Add_additions,visit)

    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)

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

    
    for epoch in range(start_epoch,num_epochs):
        # fit(epoch,model,text_recon_loss,y_bce_loss,trainloader,optimizer,flag='train')
        # fit(epoch,model,text_recon_loss,y_bce_loss,testloader,optimizer,flag='te+st')
        model = fit(epoch,model,text_recon_loss,y_bce_loss,train_dataset.__len__(),trainloader,optimizer,flag='train')
        model = fit(epoch,model,text_recon_loss,y_bce_loss,test_dataset.__len__(),testloader,optimizer,flag='test')



    







