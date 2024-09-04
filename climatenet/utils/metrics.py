import numpy as np
import torch

def get_iou_perClass(confM):
    """
    Takes a confusion matrix confM and returns the IoU per class
    """
    unionPerClass = confM.sum(axis=0) + confM.sum(axis=1) - confM.diagonal()
    iouPerClass = np.zeros(3)
    for i in range(0,3):
        if unionPerClass[i] == 0:
            iouPerClass[i] = 1
        else:
            iouPerClass[i] = confM.diagonal()[i] / unionPerClass[i]
    return iouPerClass
        
def get_cm(pred, gt, n_classes=3):
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(pred)):
        pred_tmp = pred[i].int()
        gt_tmp = gt[i].int()

        for actual in range(n_classes):
            for predicted in range(n_classes):
                is_actual = torch.eq(gt_tmp, actual)
                is_pred = torch.eq(pred_tmp, predicted)
                cm[actual][predicted] += len(torch.nonzero(is_actual & is_pred))
  
    return cm

def get_cm_avg(pred, n_classes=3, timestamps=None, timestamp_dataset=None, device=None):
    cm = np.zeros((n_classes, n_classes))
    for i in range(len(pred)):
        pred_tmp = pred[i].int()
        time = timestamps[i]
        labels = timestamp_dataset[time]

        avg_cm = np.zeros((n_classes, n_classes))

        # Go through each expert labels
        for label in labels:
            gt_tmp = torch.tensor(label.values).int().to(device)

            for actual in range(n_classes):
                for predicted in range(n_classes):
                    is_actual = torch.eq(gt_tmp, actual)
                    is_pred = torch.eq(pred_tmp, predicted)
                    avg_cm[actual][predicted] += len(torch.nonzero(is_actual & is_pred))
        
        # average the cm
        avg_cm = avg_cm / len(labels)
        cm += avg_cm
       
    return np.ceil(cm)