
import xgboost
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from matplotlib import pyplot as plt
import pickle
import operator
from sklearn import preprocessing
from sklearn.preprocessing import normalize
import os
from xgboost import plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
import dill

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
import warnings
from sklearn import metrics
from xgboost import plot_tree
from sklearn.decomposition import PCA
from collections import defaultdict
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn import svm
warnings.filterwarnings('ignore')
TEXT = dill.load(open("TEXT.Field","rb"))


# scaler = MinMaxScaler()
# train_dir = 'data/train.csv'
# test_dir = 'data/test.csv'
visit = 'twice'
class_3 = True
# parameters = {
#               'max_depth': [3,4,5],
#             #   'learning_rate': [0.2],
#               'n_estimators': [100,200,300],
#               'min_child_weight': [2,3]
# }
# model = svm.SVC(kernel="poly",max_iter = 100,verbose=True)

model = xgb.XGBClassifier(max_depth=4,
			learning_rate=0.1,
			n_estimators=100,
			nthread=-1,
			min_child_weight=3,
            scale_pos_weight = 13,
			max_delta_step=  0.01,
			seed=1440
            )
clf_multilabel = OneVsRestClassifier(model)
# gsearch = GridSearchCV(clf_multilabel, param_grid=parameters,scoring='f1_macro', cv=3,verbose=True)

random_state=0
max_length = 300

text_dir = '/home/comp/cssniu/mllt/dataset/brief_course/'
data_dir = '/home/comp/cssniu/mllt/dataset/new_packed_data/'

stopword = list(pd.read_csv('/home/comp/cssniu/RAIM/stopwods.csv').values.squeeze())
label_name_list = ["Acute and unspecified renal failure",
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
def read_data(data_list,data_dir):
    text_list = []
    label_list = []
    # print(data_list)
    for d in data_list:
        df = pd.read_csv(text_dir+"_".join(d.split("_")[:2])+".csv").values
        text = df[:,1:2].tolist()
        text = [str(i[0]) for i in text if not str(i[0]).isdigit()]
        text = ' '.join(text)
        text = rm_stop_words(text)
        text = [tok for tok in text.split(" ")] 
        text =[TEXT.vocab.stoi[t] for t in text]
        if len(text) >= max_length:
            text = text[:max_length]
        else:
            text = text + [2]*(max_length-len(text))
        text_list.append(text)
        event_df = pd.read_csv(os.path.join(data_dir,d))
        label = list(event_df[label_name_list].values[:1,:][0])
        cluster_label = [0,0,0]
        if class_3:
            if sum(label[:13]) >=1:
                cluster_label[0] = 1
            if sum(label[13:20]) >= 1:
                cluster_label[1] = 1
            if sum(label[20:]) >= 1:
                cluster_label[2] = 1
            label_list.append(cluster_label)
        else:
            label_list.append(label)
    return np.array(text_list),np.array(label_list)
def dataloader(flag):
    if flag == 'once':
        train_dir = os.path.join(data_dir,visit,"train")
        test_dir = os.path.join(data_dir,visit,"test")
    else:
        train_dir = os.path.join(data_dir,'once',"train")
        test_dir = os.path.join(data_dir,visit,"test1")
    train_list = os.listdir(train_dir)
    test_list = os.listdir(test_dir)

    X_train,y_train = read_data(train_list,train_dir)
    X_test,y_test = read_data(test_list,test_dir)
    X_train,y_train = shuffle(X_train,y_train, random_state=random_state)
    X_test,y_test = shuffle(X_test,y_test, random_state=random_state)
    return X_train,y_train,X_test,y_test

    

X_train,y_train,X_test,y_test = dataloader(visit) 

# gsearch = GridSearchCV(clf_multilabel, param_grid=parameters,scoring='f1_macro', cv=3,verbose=True)
# gsearch.fit(X_train, y_train)
# # pickle.dump(gsearch, open("xgboost.pickle.dat", "wb"))

# print("Best score: %0.3f" % gsearch.best_score_)
# print("Best parameters set:")
# best_parameters = gsearch.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_parameters[param_name]))

clf_multilabel.fit(X_train, y_train)
y_pred = clf_multilabel.predict(X_test)
precision = metrics.precision_score(y_test,y_pred,average=None)
print("precision: ",precision)
precision = metrics.precision_score(y_test,y_pred,average='macro')
print("macro precision: ",precision)
precision = metrics.precision_score(y_test,y_pred,average='micro')
print("micro precision: ",precision)

recall = metrics.recall_score(y_test,y_pred,average=None)
print("recall: ",recall)
recall = metrics.recall_score(y_test,y_pred,average='macro')
print("macro recall: ",recall)
precision = metrics.recall_score(y_test,y_pred,average='micro')
print("micro recall: ",recall)

f1_score = metrics.f1_score(y_test,y_pred,average=None)
print("f1_score :",f1_score)
f1_score = metrics.f1_score(y_test,y_pred,average="micro")
print("micro f1_score: ",f1_score)
f1_score = metrics.f1_score(y_test,y_pred,average="macro")
print("macro f1_score: ",f1_score)

roc_auc_score = metrics.roc_auc_score(y_test,y_pred,average=None)
print("roc_auc_score :",roc_auc_score)
roc_auc_score = metrics.roc_auc_score(y_test,y_pred,average="micro")
print("mairo_auc: ",roc_auc_score)
roc_auc_score = metrics.roc_auc_score(y_test,y_pred,average="macro")
print("macro_auc: ",roc_auc_score)
