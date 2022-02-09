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
from load_label_descript import label_descript

from datetime import datetime
SEED = 2019
torch.manual_seed(SEED)
class PatientDataset(object):
    def __init__(self, data_dir,max_length,class_3,visit,flag="train",):
        self.data_dir = data_dir
        self.flag = flag
        self.class_3 = class_3
        self.label_embedding = label_descript()
        self.visit = visit
        self.text_dir = '/home/comp/cssniu/mllt/dataset/brief_course/'
        self.datedf = pd.read_csv('/home/comp/cssniu/mllt/dataset/new_packed_data/date_file.csv')
        self.stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())
        self.max_length = max_length
        if visit == 'twice':
            self.patient_list = os.listdir(os.path.join(f'{data_dir}',flag+"1"))
        else:
            self.patient_list = os.listdir(os.path.join(f'{data_dir}',flag))

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
    def rm_stop_words(self,text,max_length):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            tmp = tmp[:max_length]
            text = ' '.join(tmp)
            return text
    def generate_fields(self,TEXT,Label_Descript,LABEL):
        fields = [('text', TEXT),('label_decspt',Label_Descript), ('label', LABEL)]
        examples = []
        for idx in  tqdm(range(len(self.patient_list)),desc=f"loading {self.flag} data"):

            patient_file = self.patient_list[idx]
            text_df = pd.read_csv(self.text_dir+"_".join(patient_file.split("_")[:2])+".csv").values
            text = text_df[:,1:2].tolist()
            text = [str(i[0]) for i in text if not str(i[0]).isdigit()]
            text = ' '.join(text)
            text = self.rm_stop_words(text,self.max_length)
            if self.visit == 'twice':
                label = list(pd.read_csv(os.path.join(self.data_dir,self.flag+"1",patient_file))[self.label_list].values[:1,:][0])
            else:
                label = list(pd.read_csv(os.path.join(self.data_dir,self.flag,patient_file))[self.label_list].values[:1,:][0])

            cluster_label = [0,0,0]
            if self.class_3:
                if sum(label[:13]) >=1:
                    cluster_label[0] = 1
                if sum(label[13:20]) >= 1:
                    cluster_label[1] = 1
                if sum(label[20:]) >= 1:
                    cluster_label[2] = 1
                label = cluster_label

            # print(cluster_label)
            examples.append(data.Example.fromlist([text, self.label_embedding,label], fields))
        model_data = data.Dataset(examples,fields)

        return model_data


    def __len__(self):
        return len(self.patient_list)


if __name__ == '__main__':
    TEXT = data.Field(sequential=True,tokenize=lambda x: x.split(), lower=True,batch_first=True)
    LABEL = data.Field(dtype=torch.long, use_vocab=False,batch_first=True)  
    max_length = 300
    train_dataset = PatientDataset("dataset/new_packed_data/once/",max_length,True,visit = "once",flag="train").generate_fields(TEXT,TEXT,LABEL)
    test_dataset = PatientDataset("dataset/new_packed_data/once/",max_length,True,visit = "once",flag="test").generate_fields(TEXT,TEXT,LABEL)
  
    # print('Buliding vocublary ...')
    import dill

    TEXT.build_vocab(train_dataset, vectors='glove.6B.100d')
    # dill.dump(TEXT,open("TEXT.Field","wb"))
    # TEXT = dill.load(open("TEXT.Field","rb"))

    # tokenized = [tok for tok in text.split(" ")]  #tokenize the sentence 
    # indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    # length = [len(indexed)]                                    #compute no. of words
    # tensor = torch.LongTensor(indexed)           #convert to tensor
    # tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    # print(tensor.shape)           

    # TEXT = dill.load(open("TEXT.Field","rb"))
    # text = TEXT.process(text)
    # print(text)
    # vocab = pickle.load(open( "vocab_obj.pkl",'rb') )
    # print(vocab) 
    # text = ['UnicodeDecodeError'] 
    # text = vocab(text)
    # print(text)
    train_iterator, test_iterator = data.BucketIterator.splits(
                                            (train_dataset, test_dataset), 
                                            batch_size=125,
                                            shuffle=True,
                                            sort = False)
    for i, Batch_data in enumerate(tqdm(train_iterator,desc=f"training model")):
        ld = Batch_data.label_decspt
