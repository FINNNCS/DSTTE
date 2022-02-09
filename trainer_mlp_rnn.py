from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_mlp_rnn import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
# from net_text_y_priori import mllt
from MLP_gru import MLP_RNN
import dill
import copy
from load_label_descript import label_descript
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,3"


TEXT = dill.load(open("TEXT.Field","rb"))

num_epochs = 2000
max_length = 300

BATCH_SIZE = 1
Test_batch_size = 1
pretrained = True
Freeze = False
SV_WEIGHTS = True
Add_additions = False
Logging = True
evaluation = False
if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
loss_ratio = [1,0,0]
Best_Roc = 0.6
Best_F1 = 0.4
visit = 'twice'
save_dir= "weights"
save_name = f"mlp_gru_label_embedding_{visit}_1228"
if Logging:
    
    logging_text = open(f"logs/{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)
start_epoch = 0
# weight_dir = "weights/mllt_base_bart_encoder_once_1221_epoch_14_loss_0.2631_f1_0.605_acc_0.7516.pth"

weight_dir = "weights/mlp_label_embedding_once_0104_epoch_24_loss_0.3123_f1_0.5705_acc_0.7197.pth"
# 

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

def padding_text(vec):
    max_length = max([len(i) for i in vec])
    temp = []
    for v in vec:
        if len(v) < max_length:
            paddings = [0]*(max_length-len(v))

            v = v + paddings

        temp.append(v)
    return temp


def collate_fn(data):
    
    text_list = data[0][0]
    label_list = data[0][1]
    event_codes =  data[0][2]
    return text_list,label_list,event_codes



def KL_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post,one):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1)).to(f"cuda:{Z_mean_prioir.get_device()}")
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL

def fit(epoch,model,label_descpt,text_recon_loss,y_bce_loss,data_length,dataloader,optimizer,flag='train'):
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
    # label_descpt = [l.split(" ") for l in label_descpt]
    label_descpt_token = []

    for l in label_descpt:
        tmp = []
        for w in  l.split(" "):
            tmp.append(TEXT.vocab.stoi[w])

        label_descpt_token.append(tmp)
    label_descpt_token = padding_text(label_descpt_token)
    label_descpt_token = torch.LongTensor(label_descpt_token).to(device)
    for i,(text_list,label_list,event_codes_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        visit_tensor = []
        label = torch.tensor(np.array(label_list)).to(torch.float32).to(device)

        if flag == "train":
            with torch.set_grad_enabled(True):


                for d in range(len(text_list)):

                    text = text_list[d]
                    
                    text = [tok for tok in text.split(" ")] 
                    text = [TEXT.vocab.stoi[t] for t in text]         
                    text = torch.LongTensor(text).unsqueeze(1).T[:,:max_length].to(device)     
                    event_codes = event_codes_list[d]
                    event_codes = [TEXT.vocab.stoi[t] for t in event_codes]        

                    event_codes = torch.LongTensor(event_codes).unsqueeze(1).T.to(device)  

                    Ztd= model(event_codes,text,label_descpt_token)

                    visit_tensor.append(Ztd)
                visit_tensor = torch.cat(visit_tensor,0).unsqueeze(0)
                all_hidden = model.transition_rnn(visit_tensor).squeeze(0)
                pred = model.emission(all_hidden)
                print(pred)

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
                for d in range(len(text_list)):
                    text = text_list[d]
                    
                    text = [tok for tok in text.split(" ")] 
                    text = [TEXT.vocab.stoi[t] for t in text]         
                    text = torch.LongTensor(text).unsqueeze(1).T[:,:max_length].to(device)     
                    event_codes = event_codes_list[d]
                    event_codes = [TEXT.vocab.stoi[t] for t in event_codes]         
                    event_codes = torch.LongTensor(event_codes).unsqueeze(1).T.to(device) 
                    Ztd= model(event_codes,text,label_descpt_token)
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
    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_f1,average="micro")
    f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(y_list,pred_list_f1,average="macro")
    total_loss = sum(batch_loss_list) / len(batch_loss_list)
    if Logging:
        logging_text.write('%s\n'%("PHASE：{} EPOCH : {} | Micro F1 : {} | Micro ROC ： {} | Total LOSS  : {} ".format(flag,epoch + 1, f1_micro,roc_micro, total_loss)))

    print("PHASE：{} EPOCH : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | Total LOSS  : {} ".format(flag,epoch + 1, f1_micro,f1_macro,roc_micro,roc_macro, total_loss))
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
    return model,roc_micro,roc_macro       
 


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
    vocab_size = len(TEXT.vocab)
    model = MLP_RNN(vocab_size)
    # model_rnn = model
    label_descpt = label_descript()

    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
        print("loading weight: ",weight_dir)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”

    ### freeze parameters ####
    optimizer = optim.Adam(model.parameters(True))

    if Freeze:
        for (i,child) in enumerate(model.children()):

            if i == 10:
                for param in child.parameters():
                    param.requires_grad = False
    ##########################



    text_recon_loss = nn.CrossEntropyLoss()
    y_bce_loss = nn.BCELoss()
    if evaluation:
        roc_micro_list = []
        roc_macro_list = []
        for epoch in range(5):
    
            model,roc_micro,roc_macro  = fit(epoch,model,label_descpt,text_recon_loss,y_bce_loss,test_dataset.__len__(),testloader,optimizer,flag='test')
            roc_micro_list.append(roc_micro)
            roc_macro_list.append(roc_macro)
        roc_micro_mean = np.mean(roc_micro_list)
        roc_macro_mean = np.mean(roc_macro_list)
        print(f"Micro roc mean : {roc_micro_mean} Macro roc : {roc_macro_mean}")

    else:
        for epoch in range(start_epoch,num_epochs):

            model,roc_micro,roc_macro = fit(epoch,model,label_descpt,text_recon_loss,y_bce_loss,train_dataset.__len__(),trainloader,optimizer,flag='train')
            model,roc_micro,roc_macro = fit(epoch,model,label_descpt,text_recon_loss,y_bce_loss,test_dataset.__len__(),testloader,optimizer,flag='test')


        







