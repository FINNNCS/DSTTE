import re
from typing import Tuple
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
from base_model import mllt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModel

from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,1"

tsne = TSNE(random_state=0, perplexity=10)
palette = sns.color_palette("bright",25)

palette1 = sns.color_palette("dark", 3)

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

device1 = "cuda:0" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)
start_epoch = 0
# weight_dir = "weights/mllt_base_bart_encoder_once_1221_epoch_14_loss_0.2631_f1_0.605_acc_0.7516.pth"
weight_dir = "weights/mllt_base_bart_encoder_once_1221_epoch_23_loss_0.3971_f1_0.5951_acc_0.7535.pth"

# weight_dir = "weights/mllt_transition_twice_twice_1222_epoch_885_loss_2.5294_f1_0.6093_acc_0.7512.pth"
# weight_dir = "weights/mllt_transition_pretrained_cls:kl_0.99:0.01_once_1224_epoch_12_loss_0.2791_f1_0.6037_acc_0.7571.pth"



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

def fit(model,text_list,label_list,event_codes_list,flag='train'):
    global Best_F1,Best_Roc

    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
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

    with torch.no_grad():
        for d in range(len(text_list)):

            text = text_list[d]
            label = label_list[d]
            event_codes = event_codes_list[d]
            label = torch.tensor(label).to(torch.float32).to(device)
            text = tokenizer(text, return_tensors="pt",padding=True, max_length=max_length).to(device)

            if text['input_ids'].shape[1] > max_length:
                text = clip_text(BATCH_SIZE,max_length,text,device)
            elif text['input_ids'].shape[1] < max_length:
                text = padding_text(BATCH_SIZE,max_length,text,device)

            event_list = tokenizer(event_codes, return_tensors="pt",padding=True).to(device)
            if d == 0:
                Ztd_last = Ztd_zero

            Ztd,Ztd_mean_post,Ztd_logvar_post,Ot_,Ot,pred,Ztd_mean_priori,Ztd_logvar_priori = \
            model(0,chief_comp_last,event_list,text,Ztd_last,Add_additions,flag)

            Ztd_last = Ztd_mean_post
        
            Ztd_mean_post = pd.DataFrame(np.array(Ztd_mean_post.cpu().data.tolist()))
            Ztd_mean_priori = pd.DataFrame(np.array(Ztd_mean_priori.cpu().data.tolist()))
            if d == 0:
                continue
            tsne_results.append(Ztd_mean_priori.values)
            # with open(f"tsv_file/Ztd_mean_post_{d}.tsv", 'w') as write_tsv1:
            #     write_tsv1.write(Ztd_mean_post.to_csv(sep='\t', index=False,header=False))
            # with open(f"tsv_file/Ztd_mean_priori_{d}.tsv", 'w') as write_tsv2:
            #     write_tsv2.write(Ztd_mean_priori.to_csv(sep='\t', index=False,header=False))

            # label = np.array(label.cpu().data.tolist())
            # pred = np.array(pred.cpu().data.tolist())
            # pred=(pred > 0.5) 
    tsne_results = np.stack(tsne_results).squeeze()
    # tsne_results = scaler.fit_transform(tsne_results)
    X_tsne = tsne.fit_transform(tsne_results)
    # X_tsne3 = scaler.fit_transform(X_tsne3.squeeze())
    y = [1,2,3,4,5,6]
    sns.scatterplot(X_tsne[:,0].squeeze(), X_tsne[:,1].squeeze(),c=y)
    for t in range(len(X_tsne)):
        plt.text(X_tsne[t:t+1,0].squeeze(),X_tsne[t:t+1,1].squeeze()+0.05,range(6)[t],size = "large")
    plt.savefig(f'images/tsne.jpg')

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
def load_data(data_dir):
    text_dir = '/home/comp/cssniu/mllt/dataset/brief_course/'
    event_dir = '/home/comp/cssniu/mllt/dataset/event_new/'

    visit_list = sorted(os.listdir(data_dir), key= sort_key)
    breif_course_list = []
    label_list = []
    event_list = []
    text_df = pd.read_csv(text_dir+"_".join(visit_list[0].split("_")[:2])+".csv").values
    for v in visit_list:

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
                    j = j.lower().split(" ")
                    words = []

                    for s in j:
                        if s.isalpha():
                            words.append(s)

                    j = " ".join(words)
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
    from torch.nn import functional as F
    model = mllt()
    model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
    # encoder = BartModel.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',do_lower_case=True,TOKENIZERS_PARALLELISM=True)

    # encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True)
    data_dir = "dataset/new_packed_data/twice/test/8698/"
    breif_course_list,label_list,event_list =  load_data(data_dir)
    text = breif_course_list[0]

    model.train()
    text_token = tokenizer(text,return_tensors="pt",padding=True)
    text_embedding = model.encoder(**text_token)[0]
    event = event_list[0]
    event_token =  tokenizer(event,return_tensors="pt", padding=True)
    # event_embedding = model.encoder(**event_token)[0][:,[0],:].transpose(1, 0)
    # B, Nt, E = text_embedding.shape
    # text_embedding = text_embedding / math.sqrt(E)
    # g = torch.bmm(text_embedding, event_embedding.transpose(-2, -1))

    # m = F.max_pool2d(g,kernel_size = (1,g.shape[-1])).squeeze(1)  # [b, l, 1]
    # b = torch.softmax(m, dim=1)  # [b, l, 1]
    print(event_token)

  


    







