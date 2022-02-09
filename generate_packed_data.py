from genericpath import exists
import torch
from torch import nn
from fairseq.models.bart import BARTModel
from fairseq.models.bart.hub_interface import BARTHubInterface
from typing import Optional
from fairseq import hub_utils
from fairseq.data import encoders
from typing import Any, Dict, Iterator, List
import torch
from fairseq.data import encoders
from torch import nn
import os
import re
import random
from collections import Counter,defaultdict
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
dataset_dir = "dataset/event_new/"
data_list = os.listdir(dataset_dir)

text_list = os.listdir("dataset/brief_course_old/")
patient_dic = defaultdict(list)
eposide_dic = {}

def diff(listA,listB):
    #求交集的两种方式
    retB = list(set(listA).intersection(set(listB)))
    if retB:
        return False
    else:
        return True

def process(data_list):
    for d in data_list:
        subj_id,hospital_id,eposide = d.split("_")[0],d.split("_")[1],d.split("_")[2][:-4]
        patient_dic[subj_id].append(hospital_id)
        eposide_dic[hospital_id] = eposide
process(data_list)
# print(len(data_list))
## 30981
n = 0
once_dic = defaultdict(list)
twice_dic = defaultdict(list)

for key in set(patient_dic.keys()):
    value = patient_dic[key]
    for v in value:
        f_name = key +"_"+v+"_"+eposide_dic[v]+".csv"
        once_dic[key].append(f_name)
    if len(value) >= 2:
        for v in value:
            f_name = key +"_"+v+"_"+eposide_dic[v]+".csv"
            twice_dic[key].append(f_name)
            n += 1
#             print(f_name)
#     if 2 <= value:
#         candidate_subj_list.append(key)
## 3737
twice_train_set, twice_test_set = train_test_split(list(twice_dic.keys()), test_size=0.2, random_state=42)

rest_list = list(set(list(once_dic.keys())) - set(list(twice_dic.keys())))
rest_train_set, rest_test_set = train_test_split(rest_list, test_size=0.2, random_state=42)

once_train_list = twice_train_set + rest_train_set
once_test_list = twice_test_set + rest_test_set

def sv_data(data_set,file_dic,visit = "once", flag = 'train'):
    for d in tqdm(data_set):
        csv_file = file_dic[d]
        if not os.path.exists( f"dataset/new_packed_data/{visit}/{flag}/{d}"):
            os.mkdir( f"dataset/new_packed_data/{visit}/{flag}/{d}")
            for c in csv_file:                
                # print(os.path.exists(f"{dataset_dir}/{c}"))
                shutil.copy(f"{dataset_dir}/{c}",f"dataset/new_packed_data/{visit}/{flag}/{d}/{c}")

# # 30981
# # 4540
# # 916 229

# ## 先统计 multi visit
# sv_data(once_train_list,once_dic,visit = "once",flag = 'train')
# sv_data(once_test_list,once_dic,visit = "once",flag = 'test')
# sv_data(twice_train_set,once_dic,visit = "twice",flag = 'train')
# sv_data(twice_test_set,once_dic,visit = "twice",flag = 'test')

### check
once_train_dir = "dataset/new_packed_data/once/train/"
once_test_dir = "dataset/new_packed_data/once/test/"
twice_train_dir = "dataset/new_packed_data/twice/train/"
twice_test_dir = "dataset/new_packed_data/twice/test/"

# once_train_list = [f for i in os.listdir(once_train_dir) for f in os.listdir(os.path.join(once_train_dir,i))]
# once_test_list = [f for i in os.listdir(once_test_dir) for f in os.listdir(os.path.join(once_test_dir,i))]
# twice_train_list = [f for i in os.listdir(twice_train_dir) for f in os.listdir(os.path.join(twice_train_dir,i))]
# twice_test_list = [f for i in os.listdir(twice_test_dir) for f in os.listdir(os.path.join(twice_test_dir,i))]

# print(diff(once_train_list,once_test_list))
# print(diff(once_train_list,twice_test_list))
# print(diff(twice_train_list,twice_test_list))

once_dir = "dataset/new_packed_data/twice/"
def mv_once_outside(once_dir,flag = 'train'):
    patient_list = os.listdir(once_dir + flag )
    for i in tqdm(patient_list):
        visit_list = os.listdir(os.path.join(once_dir,flag,i))
        for j in visit_list:
            visit_file_dir = os.path.join(once_dir,flag,i,j)
            if not os.path.exists( os.path.join(once_dir,flag + "1")):
                os.mkdir( os.path.join(once_dir,flag + "1"))

            shutil.copy(visit_file_dir, os.path.join(once_dir,flag + "1", j))
            print( os.path.join(once_dir,flag + "1", j))
# mv_once_outside(once_dir,flag = 'train')

def generate_multivisit(data_dir, tart_dir, flag = 'test', visit = 'thirdth'):
    patients_list = os.listdir(os.path.join(data_dir,flag))
    for p in patients_list:
        visit_list = os.listdir(os.path.join(data_dir,flag,p))
        if len(visit_list) == 2:
            if not os.path.exists(os.path.join(tart_dir,"twice")):
                os.mkdir(os.path.join(tart_dir,"twice"))
            if not os.path.exists(os.path.join(tart_dir,"twice",flag)):
                os.mkdir(os.path.join(tart_dir,"twice",flag))
            if  not os.path.exists(os.path.join(tart_dir,"twice",flag,p)):
                shutil.copytree(os.path.join(data_dir,flag,p), os.path.join(tart_dir,"twice",flag,p),symlinks=False, ignore=None)
        if len(visit_list) == 3:
            if not os.path.exists(os.path.join(tart_dir,"three")):
                os.mkdir(os.path.join(tart_dir,"three"))
            if not os.path.exists(os.path.join(tart_dir,"three",flag)):
                os.mkdir(os.path.join(tart_dir,"three",flag))
            if  not os.path.exists(os.path.join(tart_dir,"three",flag,p)):
                shutil.copytree(os.path.join(data_dir,flag,p), os.path.join(tart_dir,"three",flag,p),symlinks=False, ignore=None)
        if len(visit_list) == 4:
            if not os.path.exists(os.path.join(tart_dir,"four")):
                os.mkdir(os.path.join(tart_dir,"four"))
            if not os.path.exists(os.path.join(tart_dir,"four",flag)):
                os.mkdir(os.path.join(tart_dir,"four",flag))
            if  not os.path.exists(os.path.join(tart_dir,"four",flag,p)):
                shutil.copytree(os.path.join(data_dir,flag,p), os.path.join(tart_dir,"four",flag,p),symlinks=False, ignore=None)
        if len(visit_list) > 4:
            if not os.path.exists(os.path.join(tart_dir,"four_plus")):
                os.mkdir(os.path.join(tart_dir,"four_plus"))
            if not os.path.exists(os.path.join(tart_dir,"four_plus",flag)):
                os.mkdir(os.path.join(tart_dir,"four_plus",flag))
            if  not os.path.exists(os.path.join(tart_dir,"four_plus",flag,p)):
                shutil.copytree(os.path.join(data_dir,flag,p), os.path.join(tart_dir,"four_plus",flag,p),symlinks=False, ignore=None)

    return

data_dir = "dataset/new_packed_data/twice/"
tart_dir = "dataset/evaluation_data/"
# generate_multivisit(data_dir, tart_dir)

mv_once_outside( "dataset/evaluation_data/four_plus/",flag = 'test')
