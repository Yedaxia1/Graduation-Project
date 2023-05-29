import re
import torch
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import numpy as np

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])
    
def sensitivity_specificity(Y_test, Y_pred, num_classes=2):

    tp, fp, fn, tn = all(Y_pred, Y_test, num_classes)

    # print("tn:", tn, "tp:", tp, "fn:", fn, "fp:", fp)

    if tn[1] == 0 and fp[1] == 0:
        specificity = 0
    else:
        specificity = tn[1] / (fp[1] + tn[1])

    if tp[1] == 0 and fn[1] == 0:
        sensitivity = 0
    else:
        sensitivity = tp[1] / (tp[1] + fn[1])

    return sensitivity, specificity
    

def all(pred,label,num_classes,one_hot=False):
    if one_hot:
        pred = np.argmax(pred,axis=-1)
        label = np.argmax(label,axis=-1)
    pred_f = pred.flatten()
    label_f = label.flatten()

    cm = confusion_matrix(label_f, pred_f, np.arange(num_classes))
    tp = np.diag(cm)
    fp = np.sum(cm,axis=0) - tp
    fn = np.sum(cm,axis=1) - tp
    tn = np.full((num_classes,),np.sum(cm)) - tp - fp - fn

    return tp, fp, fn, tn
