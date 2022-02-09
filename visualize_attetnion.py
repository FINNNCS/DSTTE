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

    return temp

def load_data(data_dir):
    
    text_dir = '/home/comp/cssniu/mllt/dataset/brief_course/'
    visit_list = sorted(os.listdir(data_dir), key= sort_key)

    breif_course_list = []
    label_list = []
    cheif_complaint_list = []
    for v in visit_list:
        text_df = pd.read_csv(text_dir+"_".join(v.split("_")[:2])+".csv").values
        breif_course = text_df[:,1:2].tolist()
        breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
        text = ' '.join(breif_course)
        breif_course_list.append(text)
        # print(pd.read_csv(os.path.join(data_dir,v))[label_name].values[:1,:])
        label = list(pd.read_csv(os.path.join(data_dir,v))[label_name].values[:1,:][0])
        label_list.append(label)
    return cheif_complaint_list,breif_course_list,label_list




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
def fit(epoch,cheif_complaint_list,text_list,label_list,model,center_embedding,label_token,flag):
    global Best_F1,Best_Roc

    device = device2
    model.eval()
    model.to(device)

    center_embedding = torch.nn.Parameter(center_embedding).to(device)
    chief_comp_last = deque(maxlen=2)

    with torch.no_grad():
        v  = 1
        text = text_list[v]
        label = label_list[v]
        Ztd_zero = torch.randn((1, model.hidden_size//2)).to(device)
        Ztd_zero.requires_grad = True
        Ztd_last = Ztd_zero
        Ztd_list = [Ztd_zero]
        label_ids =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)

        label_embedding = model.encoder(**label_ids).last_hidden_state.sum(1)
        string_token = tokenizer.tokenize(text)[:300]
        print(text)

        text = tokenizer(text, return_tensors="pt",padding=True,max_length = max_length).to(device)


        if text['input_ids'].shape[1] > max_length:
            text = clip_text(BATCH_SIZE,max_length,text,device)

        if v == 0:
            Ztd_last = Ztd_zero
        phi,Center,Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
        model(center_embedding,Ztd_list,label_embedding,chief_comp_last,text,Ztd_last,flag,cluster,attention)
        # print(attention_weights,torch.max(attention_weights))
        Ztd_last = Ztd_mean_post
        weights = attention_weights[0:1,1:-1].squeeze().tolist()   
        max_num_index_weights = sorted(list(map(weights.index, heapq.nlargest(len(weights)//3, weights))))
        # print(max_num_index_weights)
        print("text 0.3: ",[string_token[i] for i in max_num_index_weights])

        y_index = np.argwhere(np.array(label) ==1 ).squeeze()
        label = [label_name[i] for i in y_index]
        print(label)


if __name__ == '__main__':
    data_dir = "dataset/new_packed_data/twice/test/8799/"
    cheif_complaint_list,breif_course_list,label_list=  load_data(data_dir)

    label_embedding = label_descript()
    center_embedding = torch.load("dataset/medical_note_embedding_kmeans_8.pth").type(torch.cuda.FloatTensor)
    model = mllt(class_3)
    model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
    fit(1,cheif_complaint_list,breif_course_list,label_list,model,center_embedding,label_embedding,flag='test')
    
    







