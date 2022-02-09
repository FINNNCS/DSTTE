import torch
from torch import nn
from torch.nn import functional as F
import math
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
import copy
# from apex import amp
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="3"
from sklearn.cluster import KMeans

device1 = "cuda:0" 


num_classes = 8
center_embedding = torch.load("dataset/medical_note_embedding_train_twice.pth",map_location=torch.device(device1)).type(torch.cuda.FloatTensor).cpu().data
clustering_model = KMeans(n_clusters=num_classes)
clustering_model.fit(center_embedding)
lables = clustering_model.labels_ + 1
centers = clustering_model.cluster_centers_ 
center_pd = pd.DataFrame(centers)

lables = pd.DataFrame(lables)

with open("tsv_file/_kmeans_data.tsv", 'w') as write_tsv:
    write_tsv.write(center_pd.to_csv(sep='\t', index=False,header=False))
with open("tsv_file/_kmeans_label.tsv", 'w') as write_tsv1:
    write_tsv1.write(lables.to_csv(sep='\t', index=False,header=False))
# print(len(os.listdir("dataset/new_packed_data/once/train/")))
# data_dir = "dataset/new_packed_data/twice/train/"
# dataset = os.listdir(data_dir)
# cnt = 0
# for p in dataset:
#     v = os.listdir(os.path.join(data_dir,p))
#     cnt += len(v)
# print(cnt)

