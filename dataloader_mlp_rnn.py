import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import re
from tqdm import tqdm
import random
from torchtext import data   
from torchtext.vocab import build_vocab_from_iterator

from datetime import datetime
SEED = 2019
torch.manual_seed(SEED)
class PatientDataset(object):
    def __init__(self, data_dir,visit,flag="train",):
        self.visit = visit
        self.data_dir = data_dir
        self.flag = flag
        self.text_dir = '/home/comp/cssniu/mllt/dataset/brief_course/'
        self.event_dir = '/home/comp/cssniu/mllt/dataset/event_new/'
        self.datedf = pd.read_csv('/home/comp/cssniu/mllt/dataset/new_packed_data/date_file.csv')
        self.stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())

        self.sbj_dir = os.path.join(f'{data_dir}',flag)
        self.sbj_list = sorted((os.listdir(self.sbj_dir)), key=lambda k: random.random()) 
        self.max_length = 1000
        self.label_list = ["Acute and unspecified renal failure",
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
    def data_processing(self,data):

        return ''.join([i.lower() for i in data if not i.isdigit()])
    def rm_stop_words(self,text):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text
    def padding_text(self,vec):
        input_ids = vec['input_ids']
        attention_mask = vec['attention_mask']
        padding_input_ids = torch.ones((input_ids.shape[0],self.max_length-input_ids.shape[1]),dtype = int).to(self.device)
        padding_attention_mask = torch.zeros((attention_mask.shape[0],self.max_length-attention_mask.shape[1]),dtype = int).to(self.device)
        input_ids_pad = torch.cat([input_ids,padding_input_ids],dim=-1)
        attention_mask_pad = torch.cat([attention_mask,padding_attention_mask],dim=-1)
        vec = {'input_ids': input_ids_pad,
        'attention_mask': attention_mask_pad}
        return vec
    def sort_key(self,text):
        temp = []
        id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
        temp.append(id_)

        return temp
    # def natural_keys(text):
    #         '''
    #     alist.sort(key=natural_keys) sorts in human order
    #     http://nedbatchelder.com/blog/200712/human_sorting.html
    #     (See Toothy's implementation in the comments)
    #     '''
    #     return [ atoi(c) for c in re.split(r'(\d+)', text)
    def __getitem__(self, idx):
        
        if self.visit == 'once':
            visit_list = [self.sbj_list[idx]]
            patient_id = self.sbj_list[idx].split("_")[0]
        else:
            patient_id = self.sbj_list[idx]
            visit_list = sorted(os.listdir(os.path.join(self.data_dir,self.flag, patient_id)), key= self.sort_key)
        breif_course_list = []
        label_list = []
        event_list = []
        text_df = pd.read_csv(self.text_dir+"_".join(visit_list[0].split("_")[:2])+".csv").values
        for v in visit_list:

            text_df = pd.read_csv(self.text_dir+"_".join(v.split("_")[:2])+".csv").values
            breif_course = text_df[:,1:2].tolist()
            breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
            text = ' '.join(breif_course)
            text = self.rm_stop_words(text)

            breif_course_list.append(text)
            event_df = pd.read_csv(self.event_dir + v)
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

                            
                        j = "".join(words)
                        # print(j)

                        # j = re.sub(r'[^a-zA-Z\s]', '', j)
                        if j in event_codes: continue
                        event_codes.append(j)
            if not event_codes:
                event_codes.append('Nan')
            event_list.append(event_codes)

            label = list(event_df[self.label_list].values[:1,:][0])
            label_list.append(label)

        return breif_course_list,label_list,event_list


    def __len__(self):
        return len(self.sbj_list)



def collate_fn(data):
    
    text_list = data[0][0]
    label_list = data[0][1]
    event_codes =  data[0][2]
    return text_list,label_list,event_codes


if __name__ == '__main__':

    max_length = 300
  
    # print('Buliding vocublary ...')
    import dill
 
    text = 'patient admitted micu intubated hypoxic respiratory transiently stopped chest ct angiogram confirmed lll pneumonia evidence antibiotic coverage included patient mechanical ventilation subsequent extubation continued continued hemodialysis micu hemodialysis catheter dislodged replaced causing delay dialysis day completed discharged' 

    TEXT = dill.load(open("TEXT.Field","rb"))

    # tokenized = [tok for tok in text.split(" ")]  #tokenize the sentence 
    # indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    # length = [len(indexed)]                                    #compute no. of words
    # tensor = torch.LongTensor(indexed)           #convert to tensor
    # tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    visit = 'twice'
    dataset = PatientDataset(f'/home/comp/cssniu/mllt/dataset/new_packed_data/{visit}/',visit,flag="train")
    batch_size = 1
    # model = cw_lstm_model(output=True)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    for i,(text_list,label,event_codes_list) in enumerate(tqdm(trainloader)):
        for d in range(len(text_list)):
            text = text_list[d]
            tokenized = [tok for tok in text.split(" ")]  
            indexed = [TEXT.vocab.stoi[t] for t in tokenized]         
            length = [len(indexed)]                                 
            tensor = torch.LongTensor(indexed)        
            tensor = tensor.unsqueeze(1).T[:,:max_length]


            print(tensor.shape)
  