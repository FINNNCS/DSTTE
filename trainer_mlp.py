from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from dataloader_mlp import *
from torchtext import data  
from sklearn import metrics
import warnings
import copy
import torch.nn.functional as F
import math
import numpy as np 

from MLP_model import MLP
SEED = 2019
torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="2,3"
num_epochs = 200
max_length = 300
class_3 = True
self_att = "cross_att"
BATCH_SIZE = 128
pretrained = False
SV_WEIGHTS = True
Logging = False
evaluation = False 
if evaluation:
    pretrained = True
    SV_WEIGHTS = False
    Logging = False
Best_Roc = 0.65
Best_F1 = 0.5
visit = 'twice'
save_dir= "weights"
save_name = f"mlp_class3_mean_{self_att}_{visit}_0205"
logging_text = open(f"logs/{save_name}.txt", 'w', encoding='utf-8')

device1 = "cuda:1" 
device1 = torch.device(device1)
device2 = "cuda:0"
device2 = torch.device(device2)
start_epoch = 0
# weight_dir = "weights/mllt_base_bart_encoder_once_1221_epoch_14_loss_0.2631_f1_0.605_acc_0.7516.pth"
# weight_dir = "weights/mlp_label_embedding_once_0104_epoch_24_loss_0.3123_f1_0.5705_acc_0.7197.pth"

# weight_dir = "weights/mlp__self_att_once_0110_epoch_167_loss_0.4047_f1_0.5209_acc_0.693.pth"

# weight_dir = "weights/mlp_label_embedding_once_0104_epoch_24_loss_0.3123_f1_0.5705_acc_0.7197.pth"
# weight_dir = "weights/mlp__no_att_once_0110_epoch_50_loss_0.3248_f1_0.5772_acc_0.7241.pth"

# weight_dir = "weights/mlp__cross_att_twice_0110_epoch_20_loss_0.3587_f1_0.5821_acc_0.7218.pth"
# weight_dir = "weights/mlp_class3_cross_att_twice_0203_epoch_52_loss_0.4452_f1_0.8746_acc_0.6312.pth"

weight_dir = "weights/mlp_class3_self_att_twice_0204_epoch_8_loss_0.4391_f1_0.8739_acc_0.62.pth"

def fit(epoch,model,y_bce_loss,dataloader,optimizer,flag='train'):
    global Best_F1,Best_Roc

    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()

    model.to(device)
    y_bce_loss.to(device)

    batch_loss_list = []
    batch_KLL_list = []
    batch_cls_list = []
    y_list = []
    pred_list_f1 = []
    for i, Batch_data in enumerate(tqdm(dataloader,desc=f"{flag}ing model")):
        if flag == "train":
            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                text_ = Batch_data.text.to(device)
                label = Batch_data.label.to(device)
                label = label.float()
                label_descpt = Batch_data.label_decspt.to(device)
                pred = model(text_,label_descpt,self_att)
                loss = y_bce_loss(pred.squeeze(),label.squeeze())
                label = np.array(label.cpu().data.tolist())
                pred = np.array(pred.cpu().data.tolist())
                pred=(pred > 0.5) 
                y_list.append(label)
                pred_list_f1.append(pred)
                loss.backward(retain_graph=True)
                optimizer.step()
                batch_loss_list.append( loss.cpu().data )  
        else:
                text_ = Batch_data.text.to(device)
                label = Batch_data.label.to(device)
                label = label.float()
                label_descpt = Batch_data.label_decspt.to(device)
                pred = model(text_,label_descpt,self_att)
                loss = y_bce_loss(pred.squeeze(),label.squeeze())
                label = np.array(label.cpu().data.tolist())
                pred = np.array(pred.cpu().data.tolist())
                pred=(pred > 0.5) 
                y_list.append(label)
                pred_list_f1.append(pred)
                batch_loss_list.append( loss.cpu().data ) 
    y_list = np.vstack(y_list)
    pred_list_f1 = np.vstack(pred_list_f1)

    precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
    recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
    precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
    recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

    f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
    roc_micro = metrics.roc_auc_score(y_list,pred_list_f1,average="micro")
    f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
    roc_macro = metrics.roc_auc_score(y_list,pred_list_f1,average="macro")
    total_loss = sum(batch_loss_list) / len(batch_loss_list)
    if Logging:
        logging_text.write('%s\n'%("PHASE：{} EPOCH : {} | Micro F1 : {} | Micro ROC ： {} | Total LOSS  : {} ".format(flag,epoch + 1, f1_micro,roc_micro, total_loss)))

    print("PHASE：{} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro, total_loss))
    if flag == 'test':
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"/home/comp/cssniu/mllt/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
            elif roc_micro > Best_Roc:
                Best_Roc = roc_micro
                PATH=f"/home/comp/cssniu/mllt/{save_dir}/{save_name}_epoch_{epoch}_loss_{round(float(loss),4)}_f1_{round(float(f1_micro),4)}_acc_{round(float(roc_micro),4)}.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)
    return model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro



















if __name__ == '__main__':
    TEXT = data.Field(sequential=True,tokenize=lambda x: x.split(), lower=True,batch_first=True)
    LABEL = data.Field(dtype=torch.long, use_vocab=False,batch_first=True)  
    max_length = 300
    train_dataset = PatientDataset(f"dataset/new_packed_data/once/",max_length,class_3,visit = "once",flag="train").generate_fields(TEXT,TEXT,LABEL)
    test_dataset = PatientDataset(f"dataset/new_packed_data/twice/",max_length,class_3,visit = "twice",flag="test").generate_fields(TEXT,TEXT,LABEL)

    train_length = train_dataset.__len__()
    test_length = test_dataset.__len__()

    print(train_length)
    print(test_length)

    print('Buliding vocublary ...')
     
    TEXT.build_vocab(train_dataset, vectors='glove.6B.100d')
    train_iterator, test_iterator = data.BucketIterator.splits(
                                            (train_dataset, test_dataset), 
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            sort = False)

    vocab_size = len(TEXT.vocab)
    print("vocab_size: ",vocab_size)
    model = MLP(class_3,vocab_size)
    
    if pretrained:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=False)
        print("loading weight: ",weight_dir)
    optimizer = optim.Adam(model.parameters(True))
    criterion = nn.BCELoss()
    if evaluation:
        precision_micro_list = []
        precision_macro_list = []
        recall_micro_list = []
        recall_macro_list = []
        f1_micro_list = []
        f1_macro_list = []
        roc_micro_list = []
        roc_macro_list = []
        for epoch in range(5):
    
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro  = fit(epoch,model,criterion,test_iterator,optimizer,flag='test')
            precision_micro_list.append(precision_micro)
            precision_macro_list.append(precision_macro)
            recall_micro_list.append(recall_micro)
            recall_macro_list.append(recall_macro)            
            f1_micro_list.append(f1_micro)
            f1_macro_list.append(f1_macro)                    
            roc_micro_list.append(roc_micro)
            roc_macro_list.append(roc_macro)
        precision_micro_mean = np.mean(precision_micro_list)
        precision_macro_mean = np.mean(precision_macro_list)        
        recall_micro_mean = np.mean(recall_micro_list)
        recall_macro_mean = np.mean(recall_macro_list)        
        f1_micro_mean = np.mean(f1_micro_list)
        f1_macro_mean = np.mean(f1_macro_list)
        roc_micro_mean = np.mean(roc_micro_list)
        roc_macro_mean = np.mean(roc_macro_list)
        print(" Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  Micro ROC : {} | Macro ROC ： {}  ".format(precision_micro_mean,precision_macro_mean,recall_micro_mean,recall_macro_mean,f1_micro_mean,f1_macro_mean,roc_micro_mean,roc_macro_mean))



    else:
        for epoch in range(start_epoch,num_epochs):
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,criterion,train_iterator,optimizer,flag='train')
            model,precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,roc_micro,roc_macro = fit(epoch,model,criterion,test_iterator,optimizer,flag='test')

    
