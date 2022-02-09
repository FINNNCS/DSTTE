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
from trasition_model import mllt

import copy
# from apex import amp

import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="2,3"
## loss traion 9:1
### ns_mllt y+recon freeze
### ns_mllt_1 y+kl freeze
### ns_mllt_2 y+kl fine_tune
### ns_mllt_3 y+recon fine_tune

## loss ration 4:6
### ns_mllt_1 y+kl fine_tune

## loss ration 7:3
### ns_mllt_2 y+kl fine_tune

### add forget gate ###
## ns_fg 6:4  forget gate
## ns_fgrnn 6:4  forget gate rnn 
## ns_mllt_3 9:1  forget gate
## ns_mllt 9:1  forget gate rnn 

### 2021 12 01
## add rnn to trasition and training fg from scatch

### 2021 12 02
## fix ztd zto not update bug gpu22 fgrnn
## add chief complaint at time stamp dimention gpu22 fg
## y only gpu22 fg

### 2021 12 03
## add chief complaint at hidden state dimention gpu22 fg

### 2021 12 06
## add time difference to prior network  gpu22 fg
## fix bug of ztd_last, and zto last from prior, which should be from post gpu22 fgrnn

### 2021 12 07
## change last hidden to the mu of post gpu22 fg
## fix zto ztd mismatch bug gpu22 fgrnn

### 2021 12 10
## base model

### 2021 12 11
## trasition model gpu24 ns_fgrnn
## trasition model with additions gpu24 ns_fg



tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)

num_epochs = 2000
max_length = 300

BATCH_SIZE = 1
Test_batch_size = 1
pretrained = True
Freeze = False
SV_WEIGHTS = False
Add_additions = False
Logging = False
loss_ratio = [0.95,0.05,0]

save_dir= "weights"
# save_name = "mllt_trasition_1211"
# if Add_additions:
#     save_name = "mllt_trasition_additions_1211"

save_name = "mllt_transition_1214"

logging_text = open("mllt_transition_1214.txt", 'w', encoding='utf-8')


device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:1" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
start_epoch = 0
weight_dir = "weights/mllt_transition_pretrain.pth"
num_of_weights = 50

once_train_dir = "dataset/new_packed_data/once/train/"
once_test_dir = "dataset/new_packed_data/once/test/"
twice_train_dir = "dataset/new_packed_data/twice/train/"
twice_test_dir = "dataset/new_packed_data/twice/test/"

once_train_list = [f for i in os.listdir(once_train_dir) for f in os.listdir(os.path.join(once_train_dir,i))]
once_test_list = [f for i in os.listdir(once_test_dir) for f in os.listdir(os.path.join(once_test_dir,i))]
twice_train_list = [f for i in os.listdir(twice_train_dir) for f in os.listdir(os.path.join(twice_train_dir,i))]
twice_test_list = [f for i in os.listdir(twice_test_dir) for f in os.listdir(os.path.join(twice_test_dir,i))]


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
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1))

        # KLD = torch.sum(0.5*(Z_logvar_post-Z_logvar_prioir+(torch.exp(Z_logvar_prioir)+(Z_mean_post-Z_mean_prioir).pow(2))/torch.exp(Z_logvar_post)-one), 1)  
        # torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)  
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL

def fit(epoch,model,text_recon_loss,y_bce_loss,data_list,dataloader,optimizer,flag='train'):
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()
    model.to(device)
    # if flag == 'train' and epoch ==0:
    #     model, optimizer = amp.initialize(model, optimizer, opt_level="O1", keep_batchnorm_fp32=True) # 这里是“欧一”，不是“零一”

    batch_loss_list = []
    batch_reconL_list = []
    chief_comp_last = deque(maxlen=2)

    batch_KLL_list = []
    batch_cls_list = []
    total_length = len(data_list)

    y_list = np.zeros((total_length,25))
    pred_list = np.zeros((total_length,25))
    l = 0
    one = torch.zeros(1).to(device)

    for i,(cheif_complaint_list,text_list,label_list,event_codes_list,time_stamp_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
     
        Ztd_zero = torch.randn((1, model.hidden_size)).to(device)
        Ztd_zero.requires_grad = True


        Kl_loss = torch.zeros(1,len(text_list)).to(device)
        cls_loss = torch.zeros(1,len(text_list)).to(device)
        Ztd_last = Ztd_zero
        # y_list = np.zeros((len(text_list),15))
        # pred_list = np.zeros((len(text_list),15))
        if flag == "train":
            with torch.set_grad_enabled(True):
                for d in range(len(text_list)):
                    l += 1

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
                        Ztd_last = Ztd_zero

                    Ztd,Ztd_mean_post,Ztd_logvar_post,Yt,Ztd_mean_priori,Ztd_logvar_priori = \
                    model(time_stamp,chief_comp_last,event_list,text,Ztd_last,Add_additions,flag)
                    Ztd_last = Ztd_mean_post



                    icd_L = y_bce_loss(Yt.squeeze(),label.squeeze())

                    if d == 0:
                        q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                    else:
                        q_ztd = kl_loss(Ztd_logvar_priori,Ztd_mean_priori,Ztd_mean_post, Ztd_logvar_post,one)

                    Kl_loss[:,d] = q_ztd
                    cls_loss[:,d] = icd_L

                    y = np.array(label.cpu().data.tolist())
                    pred = np.array(Yt.cpu().data.tolist())
                    pred=(pred > 0.5) 
                    y_list[d,:] = y
                    pred_list[d,:] = pred


                cls_loss_p = cls_loss.view(-1).mean()
                kl_loss_p = Kl_loss.view(-1).mean()

                batch_cls_list.append(cls_loss_p.cpu().data )
                batch_KLL_list.append(kl_loss_p.cpu().data )
                total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p 
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
                        Ztd_last = Ztd_zero

                    Ztd,Ztd_mean_post,Ztd_logvar_post,Yt,Ztd_mean_priori,Ztd_logvar_priori = \
                    model(time_stamp,chief_comp_last,event_list,text,Ztd_last,Add_additions,flag)

                    Ztd_last = Ztd_mean_post

                    icd_L = y_bce_loss(Yt.squeeze(),label.squeeze())
                    if d == 0:
                        q_ztd = torch.mean(-0.5 * torch.sum(1 + Ztd_logvar_post - Ztd_mean_post ** 2 - Ztd_logvar_post.exp(), dim = 1), dim = 0)

                    else:
                        q_ztd = kl_loss(Ztd_logvar_priori,Ztd_mean_priori,Ztd_mean_post, Ztd_logvar_post,one)

                    Kl_loss[:,d] = q_ztd
                    cls_loss[:,d] = icd_L

                    y = np.array(label.cpu().data.tolist())
                    pred = np.array(Yt.cpu().data.tolist())
                    pred=(pred > 0.5) 
                    y_list[d,:] = y
                    pred_list[d,:] = pred


                cls_loss_p = cls_loss.view(-1).mean()
                kl_loss_p = Kl_loss.view(-1).mean()

                batch_cls_list.append(cls_loss_p.cpu().data )
                batch_KLL_list.append(kl_loss_p.cpu().data )
                total_loss = loss_ratio[0]*cls_loss_p + loss_ratio[1]*kl_loss_p 
                loss = total_loss.cpu().data 
                batch_loss_list.append(loss)   
          
    # print(y_list,pred_list)
    f1 = metrics.f1_score(y_list,pred_list,average="micro")
    acc = metrics.roc_auc_score(y_list,pred_list,average="micro")
    total_loss = sum(batch_loss_list) / len(batch_loss_list)
    total_cls = sum(batch_cls_list) / len(batch_cls_list)
    total_kls = sum(batch_KLL_list) / len(batch_KLL_list)
    print("y: ",y)
    print("pred: ",pred)

    print("PHASE：{} EPOCH : {} | F1 : {} | ROC ： {} | Total LOSS  : {} |  Clss Loss : {} | KL Loss : {} ".format(flag,epoch + 1,  f1,acc, total_loss, total_cls,total_kls))
    if Logging:
        logging_text.write('%s\n'%("PHASE：{} EPOCH : {} | F1 : {} | ROC ： {} | Total LOSS  : {} |  Clss Loss : {} | KL Loss : {} ".format(flag,epoch + 1,  f1,acc, total_loss, total_cls,total_kls)))

    if SV_WEIGHTS:
        if epoch <= num_of_weights:
            if flag == 'test':
                PATH=f"/home/comp/cssniu/mllt_backup/mllt/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1),4)}_acc_{round(float(acc),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
    return model       


if __name__ == '__main__':
    # train_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="train")
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    # test_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="test")
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    train_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/twice/',flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    test_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/twice/',flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    print(train_dataset.__len__())
    print(test_dataset.__len__())

    model = mllt(Add_additions)

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
        model = fit(epoch,model,text_recon_loss,y_bce_loss,twice_train_dir,trainloader,optimizer,flag='train')
        model = fit(epoch,model,text_recon_loss,y_bce_loss,twice_test_dir,testloader,optimizer,flag='test')



    







