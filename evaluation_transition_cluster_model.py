from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_eval import PatientDataset
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
import re
import heapq
from random import sample
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,1"
import matplotlib.pyplot as plt
sns.set(style="whitegrid")
from sklearn.cluster import KMeans

tsne = TSNE(random_state=0,perplexity=8)
palette = sns.color_palette("Set2",9)

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
attention = "cross_att"

num_epochs = 2000
max_length = 300
Add_additions = False
class_3 = True
cluster = True
self_att = False
BATCH_SIZE = 30
# BATCH_SIZE = 60
cluster_number = 8
Freeze = True
evaluation = True
pretrained = True
SV_WEIGHTS = False
Logging = False
if self_att:
    from transition_selfatt import mllt
label_name = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]
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
# weight_dir = "weights/mllt_clinical_bert_pretrain_transition_cluster_label_att_batch_twice_0128_epoch_37_loss_0.5558_f1_0.8913_acc_0.6322.pth"

weight_dir = "weights/mllt_clinical_bert_pretrain_transition_cluster_batch_30_label_att_batch_twice_0128_epoch_18_loss_0.5444_f1_0.891_acc_0.6515.pth"
def sort_key(text):
    temp = []
    id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
    temp.append(id_)
patient_id = 47240
target_list = os.listdir(f"/home/comp/cssniu/mllt/dataset/cluster_eval_data/twice/test/{patient_id}/")
# target_list = sorted(os.listdir("/home/comp/cssniu/mllt/dataset/cluster_eval_data/twice/test/25225/"), key= sort_key)
# print(target_list)

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
    cheif_complaint_list = [d[0] for d in data]
    text_list = [d[1] for d in data]
    label_list =[d[2] for d in data]
    # cheif_complaint_list = data[0][0]
    # text_list = data[0][1]
    # label_list =data[0][2]
    return cheif_complaint_list,text_list,label_list


def KL_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1)).to(f"cuda:{Z_mean_prioir.get_device()}")
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL


def train_patients(target_list,y_list,X_tsne,visit_label,attention_list):
    patient_idx = []
    for t in target_list:
        patient_idx.append(test_dataset.file_name.index(t))
    patient_idx = sorted(patient_idx)

    patients_labels = np.array(y_list)[patient_idx[0]:patient_idx[-1]+1,:]
    patient_clusters = np.array(visit_label)[patient_idx[0]:patient_idx[-1]+1]
    patient_attentions = list(np.array(attention_list)[patient_idx[0]:patient_idx[-1]+1,:][[1],:].squeeze())
    max_num_index_weights = sorted(list(map(patient_attentions.index, heapq.nlargest(len(patient_attentions)//3, patient_attentions))))

    print(max_num_index_weights)

    label_name_list = []


    patient_embedding = X_tsne[patient_idx[0]:patient_idx[-1]+1,:]
    # sns.scatterplot(patient_embedding[:,0].squeeze(), patient_embedding[:,1].squeeze(),marker="X",s=150)
    for p in patients_labels:
        y_index = np.argwhere(np.array(p) ==1 ).squeeze()
        label_name_list.append([label_name[i] for i in y_index])
    for t in range(len(patient_embedding)):
        plt.text(patient_embedding[t:t+1,0].squeeze(),patient_embedding[t:t+1,1].squeeze()+0.05,t+1,size = "large")
    print(label_name_list)
def fit(epoch,test_dataset,model,center_embedding,label_token,text_recon_loss,y_bce_loss,cluster_loss,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc

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
    phi_list = []
    attention_list = []
    cnt = 0
    for i,(cheif_complaint_list,text_list,label_list) in enumerate(tqdm(dataloader)):
        # if i == 2: break
        optimizer.zero_grad()
        center_embedding = torch.nn.Parameter(center_embedding).to(device)
        batch_KLL_list = torch.zeros(len(text_list)).to(device)
        batch_cls_list = torch.zeros(len(text_list)).to(device)
      
        with torch.no_grad():
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
                    if sum(label) == 1:
                        cnt+=1
                    # cheif_complaint = cheif_complaint_list[d]
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
                    model(center_embedding,Ztd_list,label_embedding,chief_comp_last,text,Ztd_last,flag,cluster,attention)
                    Ztd_last = Ztd_mean_post
                    Ztd_list.append(Ztd_last)
                    phi_list.append(phi.squeeze().cpu().data.numpy())
                    attention_list.append(attention_weights.squeeze().cpu().data.numpy())
                    all_embedding.append(Ztd.squeeze().cpu().data.numpy())
                    y_list.append(label)

            # phi_batch = torch.cat(phi_list,0)
            # print(phi_batch[:5,:])
          
    phi_list = np.stack(phi_list)
    attention_list = np.stack(attention_list)

    all_embedding = np.stack(all_embedding)
  
    Center = Center.cpu().data.numpy()
    visit_label = []
    # single_
    for d in range(all_embedding.shape[0]):

        single_phi = list(phi_list[d].squeeze())
        visit_label.append(single_phi.index(max(single_phi))+1)
    # # # revert_visit_label = np.zeros(all_embedding.shape[0])
    # # # revert_visit_label[patient_idx[0]:patient_idx[-1]+1] = np.array(visit_label[patient_idx[0]:patient_idx[-1]+1])
    all_embedding_pd = pd.DataFrame(all_embedding)
    visit_label_pd = pd.DataFrame(np.array(visit_label),columns = ["label"])
    # # new_embedding = []
    # # new_label = []
    # # cluste2 = visit_label_pd[visit_label_pd["label"]==2]
    # # cluste5 = visit_label_pd[visit_label_pd["label"]==5]


    # # cluster2_index = list(set(cluste2.index.values).difference(set(sample(list(cluste2.index.values),150))))
    
    # # cluster5_index =  list(set(cluste5.index.values).difference(set(sample(list(cluste5.index.values),150))))
    # # total_index = cluster2_index + cluster5_index
    # visit_label_pd = visit_label_pd.drop(total_index)
    # all_embedding_pd = all_embedding_pd.drop(total_index)
    with open("tsv_file/train_data.tsv", 'w') as write_tsv:
        write_tsv.write(all_embedding_pd.to_csv(sep='\t', index=False,header=False))
    with open("tsv_file/train_label.tsv", 'w') as write_tsv1:
        write_tsv1.write(visit_label_pd.to_csv(sep='\t', index=False,header=False))

    # num_classes = 8
    # # center_embedding = torch.load("dataset/medical_note_embedding_train_twice.pth",map_location=torch.device(device1)).type(torch.cuda.FloatTensor).cpu().data
    # clustering_model = KMeans(n_clusters=num_classes)
    # clustering_model.fit(all_embedding)
    # lables = clustering_model.labels_ + 1
    # lables = pd.DataFrame(lables)

    # with open("tsv_file/_kmeans_label.tsv", 'w') as write_tsv2:
    #     write_tsv2.write(lables.to_csv(sep='\t', index=False,header=False))



    # # # patients_label_name = [[label_name[l] for l in p] for p in patients_labels]
    # # # print(patients_label_name)


    # # # print(revert_visit_label[patient_idx[0]:patient_idx[-1]+1])
    X_tsne = tsne.fit_transform(all_embedding)
    # # # with open(f"tsv_file/all_embedding.tsv", 'w') as write_tsv1:
    # # #     write_tsv1.write(pd.DataFrame(all_embedding).to_csv(sep='\t', index=False,header=False))
    # # # X_tsne3 = scaler.fit_transform(X_tsne3.squeeze())
    # # # y = [i for i in range(1,8+1)]
    # # # sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(),c=y)
    # # # sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze())
  
    # # plot = sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(),c=visit_label,palette = palette,s=10,hue = visit_label )
    # # # for t in range(len(X_tsne)):
    # #     plt.text(X_tsne[t:t+1,0].squeeze(),X_tsne[t:t+1,1].squeeze()+0.05,t+1,size = "large")
    train_patients(target_list,y_list,X_tsne,visit_label,attention_list)
    # # plot.legend(loc="lower right")

    # # plt.savefig(f'images/tsne_cluster_patinet.jpg')





if __name__ == '__main__':
    # train_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="train")
    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    # test_dataset = PatientDataset('/home/comp/cssniu/mllt_backup/mllt/dataset/new_packed_data/once/',flag="test")
    # testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True)
    test_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/new_packed_data/{visit}/',class_3,visit,flag="test")

    # test_dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/cluster_eval_data/{visit}/',class_3,visit,flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = False)
    label_embedding = label_descript()
    center_embedding = torch.load("dataset/medical_note_embedding_kmeans_8.pth").type(torch.cuda.FloatTensor)
    # print(test_dataset.sbj_list)
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
    
    







