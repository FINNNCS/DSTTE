import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_base import PatientDataset

import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel, BartPretrainedModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
# from net_text_y_priori import mllt
from base_cluster_two_opt_model import encoder_net,cluster_net
# from clinical_bert import mllt

from transformers import AutoTokenizer
from load_label_descript import label_descript
import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="2,3"

from transformers import AutoTokenizer, AutoModel

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)
# tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
class_3 = True
weighted = True
fused = False
train_base = False
self_att = "cross_att"
num_epochs = 2000
max_length = 300
BATCH_SIZE = 60
evaluation = False
pretrained = True
Freeze = False
SV_WEIGHTS = True
Logging = False
if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
loss_ratio = [1,1,0]
Best_Roc = 0.5
Best_F1 = 0.2
visit = 'once'
save_dir= "weights"
save_name = f"basemodel_cluster_10_pretrained_3_weighted_ct_two_opt_{self_att}_{visit}_0126"
if Logging:
    logging_text = open(f"{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:0" if torch.cuda.is_available() else "cpu"
device1 = torch.device(device1)
device2 = "cuda:1" if torch.cuda.is_available() else "cpu"
device2 = torch.device(device2)
start_epoch = 0
# weight_dir = "weights/mllt_base_bart_encoder_once_1221_epoch_23_loss_0.3971_f1_0.5951_acc_0.7535.pth"
# weight_dir = "weights/mllt_base_cross_att_once_0109_epoch_18_loss_0.3256_f1_0.6063_acc_0.7499.pth"
# weight_dir = "weights/mllt_base_no_att_once_0109_epoch_22_loss_0.3152_f1_0.6044_acc_0.7449.pth"
# weight_dir = "weights/mllt_base_self_att_once_0109_epoch_18_loss_0.3281_f1_0.6039_acc_0.7478.pth"
# weight_dir = "weights/basemodel_cluster_initial_cross_att_once_0118_epoch_59_loss_0.2607_f1_0.6013_acc_0.7503.pth"
# weight_dir = "weights/basemodel_cluster_pretrained_y_cluster_fused_cross_att_once_0120_epoch_8_loss_0.4431_f1_0.5423_acc_0.7063.pth"
weight_dir = "weights/basemodel_cluster_pretrained_class3_cross_att_once_0125_epoch_4_loss_0.4309_f1_0.8429_acc_0.7452.pth"
# weight_dir = "weights/basemodel_cluster_10_pretrained_3_no_weighted_ct_cross_att_once_0123_epoch_3_loss_30.6667_f1_0.7988_acc_0.5.pth"
# weight_dir = "weights/basemodel_cluster_10_pretrained_3_weighted_ct_cross_att_once_0123_epoch_3_loss_0.6082_f1_0.8209_acc_0.6554.pth"
# weight_dir = "weights/basemodel_cluster_10_pretrained_3_weighted_ct_cross_att_once_0125_epoch_0_loss_0.7423_f1_0.7981_acc_0.5.pth"

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

# def collate_fn(data):
    
#     cheif_complaint_list = data[0][0]
#     text_list = data[0][1]
#     label_list = data[0][2]
#     event_codes =  data[0][3]
#     time_stamp_list = data[0][4]
#     return cheif_complaint_list,text_list,label_list,event_codes,time_stamp_list

def collate_fn(data):
    
    cheif_complaint_list = [d[0] for d in data]
    text_list = [d[1][0] for d in data]
    label_list = [d[2] for d in data]
    event_codes =  [d[3][0] for d in data]
    time_stamp_list =[d[4] for d in data]
    return cheif_complaint_list,text_list,label_list,event_codes,time_stamp_list


def kl_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post,one):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1))

        # KLD = torch.sum(0.5*(Z_logvar_post-Z_logvar_prioir+(torch.exp(Z_logvar_prioir)+(Z_mean_post-Z_mean_prioir).pow(2))/torch.exp(Z_logvar_post)-one), 1)  
        # torch.sum(0.5*(logvar2-logvar1+(torch.exp(logvar1)+(mu1-mu2).pow(2))/torch.exp(logvar2)-one), 1)  
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL

# def label_loss(L_E,criterion, y):
#     """
#     torch.tensor([0,1,2]) is decoded identity label vector
#     """ 
   
#     y_c =  torch.range(0, y.shape[1] - 1, dtype=torch.long).unsqueeze(0).to(f"cuda:{L_E.get_device()}")
#     print(L_E.shape,y_c.shape)
#     return criterion(L_E,y_c)

def fit(epoch,model_encoder,model_cluster,center_embedding,label_embedding,cluster_loss,y_cluster_bce_loss,test_length,dataloader,optimizer_encoder,optimizer_cluster,flag='train'):
    global Best_F1,Best_Roc
    if flag == 'train':
        device = device1
        model_encoder.train()
        model_cluster.train()

    else:
        device = device2
        model_encoder.eval()
        model_cluster.eval()

    model_encoder.to(device)
    model_cluster.to(device)
    cluster_loss.to(device)
    y_cluster_bce_loss.to(device)
    center_embedding = torch.nn.Parameter(center_embedding).to(device)
    # center_embedding = center_embedding.to(device)
    # center_embedding.require_grads = True
    batch_loss_list = []
    batch_cls_loss = []
    batch_cls_pred_loss = []
    batch_cluster_loss = []

    y_list = []
    pred_list_f1 = []
    l = 0

    for i,(cheif_complaint_list,text_list,label_list,event_codes_list,time_stamp_list) in enumerate(tqdm(dataloader)):
        optimizer_encoder.zero_grad()
        optimizer_cluster.zero_grad()

        if flag == "train":
            with torch.set_grad_enabled(True):
                

                label = torch.tensor(label_list).to(torch.float32).squeeze(1).to(device)

                text = tokenizer(text_list, return_tensors="pt",padding=True,max_length = max_length).to(device)

                if text['input_ids'].shape[1] > max_length:
                    text = clip_text(BATCH_SIZE,max_length,text,device)
                elif text['input_ids'].shape[1] < max_length:
                    text = padding_text(BATCH_SIZE,max_length,text,device)
                # event_list = tokenizer(event_codes_list, return_tensors="pt",padding=True).to(device)
                # print(event_codes_list)
                event_list = [tokenizer(e, return_tensors="pt",padding=True).to(device) for e in event_codes_list]
                event_list = padding_event(event_list)
                label_token =  tokenizer(label_embedding, return_tensors="pt",padding=True,max_length = max_length).to(device)
                Yt_predictor,ztd =  model_encoder(text,label_token,self_att)
                Ct,phi,cluster_target = model_cluster(ztd,center_embedding,weighted)
                if fused:
                    Yt =  model_encoder.sigmoid(model_encoder.drop_out(model_encoder.MLPs(model_encoder.drop_out(model_encoder.fuse_fc(torch.cat((Ct,ztd),-1))))))
                else:
                    Yt =  model_encoder.sigmoid(model_encoder.drop_out(model_encoder.MLPs(Ct)))
                    # print(Yt)
                cls_pred_loss = y_cluster_bce_loss(Yt_predictor.squeeze(),label.squeeze())
                cls_loss = y_cluster_bce_loss(Yt.squeeze(),label.squeeze())
                cluster_lss = cluster_loss((phi+1e-08).log(),cluster_target)/phi.shape[0]

                loss = loss_ratio[0]*cls_loss + loss_ratio[1]*cluster_lss + loss_ratio[2]*cls_pred_loss
                batch_cls_pred_loss.append(cls_pred_loss.cpu().data)

                batch_cls_loss.append(cls_loss.cpu().data)
                batch_cluster_loss.append(cluster_lss.cpu().data)
                pred = np.array(Yt.cpu().data.tolist())
                y = np.array(label.cpu().data.tolist())
                pred=(pred > 0.5) 
                
                y_list.append(y)
                pred_list_f1.append(pred)
                loss.backward(retain_graph=True)
                optimizer_encoder.step()
                optimizer_cluster.step()

                batch_loss_list.append( loss.cpu().data )  
                l+=1

        else:
            with torch.no_grad():

                label = torch.tensor(label_list).to(torch.float32).squeeze(1).to(device)

                text = tokenizer(text_list, return_tensors="pt",padding=True,max_length = max_length).to(device)

                if text['input_ids'].shape[1] > max_length:
                    text = clip_text(BATCH_SIZE,max_length,text,device)
                elif text['input_ids'].shape[1] < max_length:
                    text = padding_text(BATCH_SIZE,max_length,text,device)

                event_list = [tokenizer(e, return_tensors="pt",padding=True).to(device) for e in event_codes_list]
                event_list = padding_event(event_list)
                label_token =  tokenizer(label_embedding, return_tensors="pt",padding=True,max_length = max_length).to(device)
                Yt_predictor,ztd =  model_encoder(text,label_token,self_att)
                Ct,phi,cluster_target = model_cluster(ztd,center_embedding,weighted)
                if fused:
                    Yt =  model_encoder.sigmoid(model_encoder.drop_out(model_encoder.MLPs(model_encoder.drop_out(model_encoder.fuse_fc(torch.cat((Ct,ztd),-1))))))
                else:
                    Yt =  model_encoder.sigmoid(model_encoder.drop_out(model_encoder.MLPs(Ct)))
                print(Yt)
                cls_pred_loss = y_cluster_bce_loss(Yt_predictor.squeeze(),label.squeeze())
                cls_loss = y_cluster_bce_loss(Yt.squeeze(),label.squeeze())
                cluster_lss = cluster_loss((phi+1e-08).log(),cluster_target)/phi.shape[0]
                loss = loss_ratio[0]*cls_loss + loss_ratio[1]*cluster_lss + loss_ratio[2]*cls_pred_loss
                batch_cls_pred_loss.append(cls_pred_loss.cpu().data)
                batch_cls_loss.append(cls_loss.cpu().data)
                batch_cluster_loss.append(cluster_lss.cpu().data)
                pred = np.array(Yt.cpu().data.tolist())

                y = np.array(label.cpu().data.tolist())
                pred=(pred > 0.5) 
                
                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append(loss.cpu().data )  
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
    total_clssfy_pred_loss = sum(batch_cls_pred_loss) / len(batch_cls_pred_loss)

    total_clssfy_loss = sum(batch_cls_loss) / len(batch_cls_loss)
    total_cluster_loss = sum(batch_cluster_loss) / len(batch_cluster_loss)
    if Logging:
        logging_text.write('%s\n'%("PHASE：{} EPOCH : {} | Micro F1 : {} | Micro ROC ： {} | Total LOSS  : {} ".format(flag,epoch + 1, f1_micro,roc_micro, total_loss)))
  
    print("PHASE：{} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | CLS LOSS  : {} | CLS Predctor LOSS  : {} | Cluster LOSS  : {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro,total_clssfy_loss, total_clssfy_pred_loss,total_cluster_loss,total_loss))

    if flag == 'test':
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"/home/comp/cssniu/mllt/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                torch.save({
                            'model_encoder':copy.deepcopy( model_encoder.state_dict()),
                            'model_cluster': copy.deepcopy(model_cluster.state_dict()),
                            'optimizer_encoder': copy.deepcopy(optimizer_encoder.state_dict()),
                            'optimizer_cluster': copy.deepcopy(optimizer_cluster.state_dict()),
                            }, PATH)
      
            elif roc_micro > Best_Roc:
                Best_Roc = roc_micro
                PATH=f"/home/comp/cssniu/mllt/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                torch.save({
                'model_encoder':copy.deepcopy( model_encoder.state_dict()),
                'model_cluster': copy.deepcopy(model_cluster.state_dict()),
                'optimizer_encoder': copy.deepcopy(optimizer_encoder.state_dict()),
                'optimizer_cluster': copy.deepcopy(optimizer_cluster.state_dict()),
                }, PATH)
    return model_encoder,model_cluster,roc_micro,roc_macro       


if __name__ == '__main__':
    eval_visit = "four"
    label_embedding = label_descript()
    center_embedding = torch.load("dataset/medical_note_embedding_kmeans_8.pth").type(torch.cuda.FloatTensor)

    train_dataset = PatientDataset(f"dataset/new_packed_data/{visit}/", class_3 = class_3, visit = visit, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f"dataset/new_packed_data/{visit}/",class_3 = class_3, visit = visit, flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)

    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()

    print(train_length)
    print(test_length)

    model_encoder = encoder_net(class_3)
    model_cluster = cluster_net()
    optimizer_encoder = optim.Adam(model_encoder.parameters(True), lr = 1e-5)
    optimizer_cluster = optim.Adam(model_cluster.parameters(True), lr = 1e-6)
    if pretrained:
        print(f"loading weights: {weight_dir}")
        # model_encoder.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2))['model_encoder'], strict=False)
        # model_cluster.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2))['model_cluster'], strict=False)
        model_encoder.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
        model_cluster.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1") # 这里是“欧一”，不是“零一”

    ### freeze parameters ####

    if Freeze:
        for (i,child) in enumerate(optimizer_encoder.children()):
            if i == 15:
                for param in child.parameters():
                    param.requires_grad = False
    ##########################


    cluster_loss = nn.KLDivLoss(reduction='sum')
    text_recon_loss = nn.CrossEntropyLoss()
    y_cluster_bce_loss = nn.BCELoss()

    regularization_loss = nn.CrossEntropyLoss()
    if evaluation:
        roc_micro_list = []
        roc_macro_list = []
        for epoch in range(5):
    
            model_encoder,model_cluster,roc_micro,roc_macro  = fit(epoch,model_encoder,model_cluster,center_embedding,label_embedding,cluster_loss,y_cluster_bce_loss,test_length,testloader,optimizer_encoder,optimizer_cluster,flag='test')
            roc_micro_list.append(roc_micro)
            roc_macro_list.append(roc_macro)
        roc_micro_mean = np.mean(roc_micro_list)
        roc_macro_mean = np.mean(roc_macro_list)
        print(f"Micro roc mean : {roc_micro_mean} Macro roc : {roc_macro_mean}")

    else:
        for epoch in range(start_epoch,num_epochs):
            # fit(epoch,model,text_recon_loss,y_bce_loss,trainloader,optimizer,flag='train')
            # fit(epoch,model,text_recon_loss,y_bce_loss,testloader,optimizer,flag='test')
            model_encoder,model_cluster,roc_micro,roc_macro  = fit(epoch,model_encoder,model_cluster,center_embedding,label_embedding,cluster_loss,y_cluster_bce_loss,train_length,trainloader,optimizer_encoder,optimizer_cluster,flag='train')
            model_encoder,model_cluster,roc_micro,roc_macro  = fit(epoch,model_encoder,model_cluster,center_embedding,label_embedding,cluster_loss,y_cluster_bce_loss,test_length,testloader,optimizer_encoder,optimizer_cluster,flag='test')



   

 







