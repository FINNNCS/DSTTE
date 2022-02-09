import os
import pandas as pd
import numpy as np
import re
from collections import Counter
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# dataset_dir = "/home/comp/cssniu/RAIM/benchmark_data/all/"
# text_list = os.listdir(os.path.join(dataset_dir,'text'))
# count = []
# for t in text_list:
#     try:    
#         id_ = re.findall(r'\d+_',t)[0][:-1]
#     except:pass
#     # print(id_)
#     count.append(id_)
# patients_list = []
# count = Counter(count)

# for k,v in count.items():
#     if v >= 3:
#         patients_list.append(k)


# x_train, x_test, y_train, y_test = train_test_split(patients_list, patients_list, test_size = 0.3)
# print(len(x_test),len(x_train))
# # 347 807
# train_patients_file_list = []

# for n in tqdm(x_train):
#     for l in text_list:
#         if l == 'train' or l=='test':continue
#         if n == re.findall(r'\d+_',l)[0][:-1]:
#             train_patients_file_list.append(l)

# test_patients_file_list = []
# for n in tqdm(x_test):
#     for l in text_list:
#         if l == 'train' or l=='test':continue
#         if n == re.findall(r'\d+_',l)[0][:-1]:
#             test_patients_file_list.append(l)

def generate_data():
    text_dir = os.path.join(dataset_dir,'text')
    ts_dir = os.path.join(dataset_dir,'data')
    event_dir = os.path.join(dataset_dir,'event1')

    for p in tqdm(train_patients_file_list):
            shutil.copy(os.path.join(event_dir,p),os.path.join('/home/comp/cssniu/mllt/dataset/event/train/',p))
            shutil.copy(os.path.join(text_dir,p),os.path.join('/home/comp/cssniu/mllt/dataset/text/train/',p))
            shutil.copy(os.path.join(ts_dir,p),os.path.join('/home/comp/cssniu/mllt/dataset/timeseries/train/',p))
    for p in tqdm(test_patients_file_list):
            shutil.copy(os.path.join(event_dir,p),os.path.join('/home/comp/cssniu/mllt/dataset/event/test/',p))
            shutil.copy(os.path.join(text_dir,p),os.path.join('/home/comp/cssniu/mllt/dataset/text/test/',p))
            shutil.copy(os.path.join(ts_dir,p),os.path.join('/home/comp/cssniu/mllt/dataset/timeseries/test/',p))

# generate_data()

## get icd code ###
def generate_label(data_dir):
    data_list = os.listdir(data_dir)
    for d in data_list:
        data = pd.read_csv(os.path.join(data_dir,d))
        print(data)

    # return 
def pack_data(data_dir,target_dir):
    data_list = os.listdir(data_dir)
    patient_list = []
    id_dic = {}
    for d in data_list:

        id_list = re.findall(r'\d+_',d)[0][:-1]
        patient_list.append(id_list)
    for i in set(patient_list):
        temp = []
        for d in data_list:
            id = re.findall(r'\d+_',d)[0][:-1]
            if i == id:
                temp.append(d)
        id_dic[i] = temp
    for k in id_dic.keys():
        directory = os.path.join(target_dir,k)
        if not os.path.exists(directory):
            os.makedirs(directory)
        for v in id_dic[k]:
            print(k,v)
            shutil.copy(os.path.join(data_dir,v),os.path.join(target_dir,k,v))


     



# train_dir = '/home/comp/cssniu/mllt/dataset/text/train/'
# test_dir = '/home/comp/cssniu/mllt/dataset/text/test/'
# all_diagnosis = '/home/comp/cssniu/mimic3-benchmarks1/data/root/all_diagnoses.csv'
# # train_list = os.listdir(train_dir)
# all_diagnosis = pd.read_csv(all_diagnosis,low_memory=True)
# generate_label(train_dir)
# print(all_diagnosis)
pack_data('/home/comp/cssniu/mllt/dataset/event/train/','/home/comp/cssniu/mllt/dataset/packed_data/event/train')
