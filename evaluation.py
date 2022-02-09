import re
from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
from transformers import BartModel, BartPretrainedModel,BartTokenizer
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
# from net_text_y_priori import mllt
# from transition_recon_model_ztd import mllt
from trasition_model import mllt
import heapq
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from load_label_descript import label_descript

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')

tsne = TSNE(random_state=0, perplexity=10)
palette = sns.color_palette("bright",25)

palette1 = sns.color_palette("dark", 3)
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
evaluation = True
if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
loss_ratio = [0.99,0.01,0]
Best_Roc = 0.7
Best_F1 = 0.6
visit = 'twice'
save_dir= "weights"
save_name = f"mllt_transition_pretrained_additions_{visit}_1226"
logging_text = open(f"logs/{save_name}.txt", 'w', encoding='utf-8')

device = "cuda:1" 
device = torch.device(device)

start_epoch = 0
# weight_dir = "weights/mllt_base_bart_encoder_once_1221_epoch_14_loss_0.2631_f1_0.605_acc_0.7516.pth"
weight_dir = "weights/mllt_transition_label_embedding_twice_0102_epoch_42_loss_0.4577_f1_0.6091_acc_0.7531.pth"

# weight_dir = "weights/mllt_transition_twice_twice_1222_epoch_885_loss_2.5294_f1_0.6093_acc_0.7512.pth"
# weight_dir = "weights/mllt_transition_pretrained_cls:kl_0.99:0.01_once_1224_epoch_12_loss_0.2791_f1_0.6037_acc_0.7571.pth"
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
def is_subtoken(word):
    if word[:2] == "##":
        return True
    else:
        return False

def detoeken(tokens):
    restored_text = []
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
            restored_text.append(tokens[i] + tokens[i+1][2:])
            if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                restored_text[-1] = restored_text[-1] + tokens[i+2][2:]
        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])
    return restored_text
def decode_attetnion(restored_original_text,tokens,score_index):
    
    restored_text = []
    tokens = [tokens[i] for i in score_index]
    for i in range(len(tokens)):
        if not is_subtoken(tokens[i]) and (i+1)<len(tokens) and is_subtoken(tokens[i+1]):
            w = tokens[i] + tokens[i+1][2:]
            if w in restored_original_text:
                restored_text.append(w)

        if (i+2)<len(tokens) and is_subtoken(tokens[i+2]):
                w = restored_text[-1] + tokens[i+2][2:]
                if w in restored_original_text:
                    restored_text[-1] = w

        elif not is_subtoken(tokens[i]):
            restored_text.append(tokens[i])

    return restored_text
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
    event_codes =  data[0][3]
    time_stamp_list = data[0][4]
    return cheif_complaint_list,text_list,label_list,event_codes,time_stamp_list



def KL_loss(Z_mean_prioir, Z_logvar_prioir,Z_mean_post,Z_logvar_post,one):
        KLD = 0.5 * torch.mean(torch.mean(Z_logvar_post.exp()/Z_logvar_prioir.exp() + (Z_mean_post - Z_mean_prioir).pow(2)/Z_logvar_prioir.exp() + Z_logvar_prioir - Z_logvar_post - 1, 1)).to(f"cuda:{Z_mean_prioir.get_device()}")
        return KLD

def reconstrution_loss(text_recon_loss, Ot_,label):
    RL = text_recon_loss(Ot_.view(-1,Ot_.shape[-1]),label.unsqueeze(-1).view(-1))
    return RL

def fit(model,label_token,text_list,label_list,event_codes_list,flag='train'):
    global Best_F1,Best_Roc

    model.eval()
    model.to(device)

    chief_comp_last = deque(maxlen=2)
    batch_loss_list = []

    batch_KLL_list = []
    batch_cls_list = []
    y_list = []
    pred_list_f1 = []
    l = 0
    one = torch.zeros(1).to(device)

   
    Ztd_zero = torch.randn((1, model.hidden_size)).to(device)
    Ztd_zero.requires_grad = True

    Ztd_last = Ztd_zero
    tsne_results = []
    print_id = 7
    with torch.no_grad():
        d = 1


        text = text_list[d]
        label = label_list[d]
        event_codes = event_codes_list[d]
        # label = torch.tensor(label).to(torch.float32).to(device)
        string_token = tokenizer.tokenize(text)[:298]
        # if d == print_id:
        print(text)
        print("...................")

        text = tokenizer(text, return_tensors="pt",padding=True, max_length=max_length).to(device)
        if text['input_ids'].shape[1] > max_length:
            text = clip_text(BATCH_SIZE,max_length,text,device)
        # elif text['input_ids'].shape[1] < max_length:
        #     text = padding_text(BATCH_SIZE,max_length,text,device)
        label_embedding =  tokenizer(label_token, return_tensors="pt",padding=True,max_length = max_length).to(device)

        event_list = tokenizer(event_codes, return_tensors="pt",padding=True).to(device)
        if d == 0:
            Ztd_last = Ztd_zero
        ztd_list = [Ztd_zero]
        Ztd,Ztd_mean_post,Ztd_logvar_post,pred,Ztd_mean_priori,Ztd_logvar_priori,attention_weights = \
        model(ztd_list,label_embedding,0,chief_comp_last,event_list,text,Ztd_last,Add_additions,flag)

        Ztd_last = Ztd_mean_post
    
        Ztd_mean_post = pd.DataFrame(np.array(Ztd_mean_post.cpu().data.tolist()))
        Ztd_mean_priori = pd.DataFrame(np.array(Ztd_mean_priori.cpu().data.tolist()))
        # pca.fit(Ztd_mean_post.values)

        tsne_results.append(Ztd_mean_post.values)
        # print(string_token)
        weights = attention_weights[0:1,1:-1].squeeze().tolist()   
        # if d ==print_id:
        max_num_index_weights = sorted(list(map(weights.index, heapq.nlargest(len(weights)//10, weights))))
        # if d == print_id:

        #     print("text 0.1: ",[string_token[i] for i in max_num_index_weights])

        max_num_index_weights = sorted(list(map(weights.index, heapq.nlargest(len(weights)//3, weights))))
        # if d == print_id:

        print("text 0.3: ",[string_token[i] for i in max_num_index_weights])
        max_num_index_weights = sorted(list(map(weights.index, heapq.nlargest(len(weights)//2, weights))))
        # print("text 0.5: ",[string_token[i] for i in max_num_index_weights])
        y_index = np.argwhere(np.array(label) ==1 ).squeeze()
        label = [label_name[i] for i in y_index]
        # if d ==print_id:
        # print(label)
        # break

        # with open(f"tsv_file/Ztd_mean_post_{d}.tsv", 'w') as write_tsv1:
        #     write_tsv1.write(pd.DataFrame(scaler.fit_transform(Ztd_mean_post)).to_csv(sep='\t', index=False,header=False))
        # with open(f"tsv_file/Ztd_mean_priori_{d}.tsv", 'w') as write_tsv2:
        #     write_tsv2.write(Ztd_mean_priori.to_csv(sep='\t', index=False,header=False))

        # label = np.array(label.cpu().data.tolist())
        # pred = np.array(pred.cpu().data.tolist())
        # pred=(pred > 0.5) 
    # tsne_results = np.stack(tsne_results).squeeze()
    # with open(f"tsv_file/Ztd_mean_post.tsv", 'w') as write_tsv1:
    #     write_tsv1.write(pd.DataFrame(scaler.fit_transform(tsne_results)).to_csv(sep='\t', index=False,header=False))
    # # tsne_results = scaler.fit_transform(tsne_results)
    # X_tsne = tsne.fit_transform(tsne_results)
    # # X_tsne3 = scaler.fit_transform(X_tsne3.squeeze())
    # y = [i for i in range(1,len(text_list)+1)]
    # sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(),c=y)
    # for t in range(len(X_tsne)):
    #     plt.text(X_tsne[t:t+1,0].squeeze(),X_tsne[t:t+1,1].squeeze()+0.05,y[t],size = "large")
    # plt.savefig(f'images/tsne.jpg')

    # for t in range(len(X_tsne2)):
    #     plt.text(X_tsne2[t:t+1,0].squeeze(),X_tsne2[t:t+1,1].squeeze()-0.05,range(25)[t])

    # sns.scatterplot(X_tsne3[:,0].squeeze(), X_tsne3[:,1].squeeze())
    # # for i in range(17):
    # #     plt.scatter(X_tsne3[:, 0], X_tsne3[:, 1],color = "orange")
    # for t in range(len(X_tsne3)):
    #     plt.text(X_tsne3[t:t+1,0].squeeze(),X_tsne3[t:t+1,1].squeeze()+0.05,range(17)[t],size = "large")
    #     plt.savefig(f'LDAM/images/tsne_label_task.jpg')

    return model      
def sort_key(text):
    temp = []
    id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
    temp.append(id_)

    return temp
stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())
def rm_stop_words(text):
            tmp = text.split(" ")
            for t in stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text

def load_data(data_dir):
    text_dir = '/home/comp/cssniu/mllt/dataset/brief_course/'
    event_dir = '/home/comp/cssniu/mllt/dataset/event_new/'

    visit_list = sorted(os.listdir(data_dir), key= sort_key)
    breif_course_list = []
    label_list = []
    event_list = []
    # print（）
    for v in visit_list:

        text_df = pd.read_csv(text_dir+"_".join(v.split("_")[:2])+".csv").values

        breif_course = text_df[:,1:2].tolist()
        breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
        text = ' '.join(breif_course)
        # text = rm_stop_words(text)
        breif_course_list.append(text)
        event_df = pd.read_csv(event_dir + v)
        event_file = event_df[event_df.columns[1:4]].values
        event_codes = []

        for i in range((len(event_file))):
            e = event_file[i]
            for j in e: 
                if not pd.isnull(j):
                    j = j.lower()
                    words = []
                    for s in j:
                        if s.isalpha():
                            words.append(s)

                        
                    j = " ".join(words)
                    # print(j)

                    # j = re.sub(r'[^a-zA-Z\s]', '', j)
                    if j in event_codes: continue
                    event_codes.append(j)
        if not event_codes:
            event_codes.append('Nan')
        event_list.append(event_codes)
        label = list(event_df[label_name].values[:1,:][0])
        label_list.append(label)
    return breif_course_list,label_list,event_list



if __name__ == '__main__':
    data_dir = "dataset/new_packed_data/twice/test/8799/"
    breif_course_list,label_list,event_list =  load_data(data_dir)
    label_embedding = label_descript()

    model = mllt(Add_additions,visit)

    model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device)), strict=False)
    # print("loading weight: ",weight_dir)
  

    model = fit(model,label_embedding,breif_course_list,label_list,event_list,flag='test')
 



    







