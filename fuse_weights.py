import torch
from collections import OrderedDict
import copy
def Merge(dict1, dict2): 
    return(dict2.update(dict1)) 
# weight_dir = "weights/mllt_transition_twice_twice_1222_epoch_885_loss_2.5294_f1_0.6093_acc_0.7512.pth"
weight_dir1 = "weights/mllt_clinical_bert_transition_label_att_twice_0126_epoch_3_loss_0.1907_f1_0.8587_acc_0.6808.pth"

# weight_dir = "weights/mllt_transition_label_embedding_twice_0102_epoch_42_loss_0.4577_f1_0.6091_acc_0.7531.pth"
# weight_dir = "weights/mllt_transition_label_embedding_twice_0102_epoch_0_loss_0.4336_f1_0.5999_acc_0.7446.pth"
# weight_dir = "weights/mllt_base_label_attention_once_0102_epoch_7_loss_0.3449_f1_0.5928_acc_0.7522.pth"
# 
weight_dir2  = "weights/basemodel_clinical_bert_cluster_10_pretrained_3_weighted_ct_sharefc_cross_att_once_0126_epoch_1_loss_0.5806_f1_0.8209_acc_0.6566.pth"

# weight_dir = "weights/mllt_transition_self_att_twice_0106_epoch_20_loss_0.3398_f1_0.6061_acc_0.7477.pth"
w1 = torch.load(weight_dir1)
w2 = torch.load(weight_dir2)
w3 = copy.deepcopy(w2)
for keys,values in w2.items():
    if "cluster_fc" in keys:
        pass
    else:
        # print(keys,values)
        del w3[keys]
# print(w4.keys())
# torch.save(w4,  "weights/mllt_clinical_bert_pretrainer_class3_fusedweighted.pth")
# w5 = torch.load("weights/mllt_clinical_bert_pretrainer_class3_fusedweighted.pth")
for keys,values in w3.items():
    w1[keys] = values
torch.save(w1,  "weights/mllt_clinical_bert_pretrainer_class3_fusedweighted.pth")
w5 = torch.load("weights/mllt_clinical_bert_pretrainer_class3_fusedweighted.pth")
print(w5.keys())
