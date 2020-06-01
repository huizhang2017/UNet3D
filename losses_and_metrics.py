import torch
import numpy as np

def weighting_DSC(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [batch, n_classes, x, y, z] probability
        y_true [batch, n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    mdsc = 0.0
    n_classes = y_pred.shape[1] # for PyTorch data format
    
    # convert probability to one-hot code    
    max_idx = torch.argmax(y_pred, dim=1, keepdim=True)
    one_hot = torch.zeros_like(y_pred)
    one_hot.scatter_(1, max_idx, 1)

    for c in range(0, n_classes):
        pred_flat = one_hot[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
        w = class_weights[c]/class_weights.sum()
        mdsc += w*((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth))
        
    return mdsc

   
def Generalized_Dice_Loss(y_pred, y_true, class_weights, smooth = 1.0):
    '''
    inputs:
        y_pred [batch, n_classes, x, y, z] probability
        y_true [batch, n_classes, x, y, z] one-hot code
        class_weights
        smooth = 1.0
    '''
    smooth = 1.
    loss = 0.
    n_classes = y_pred.shape[1]
    
    for c in range(0, n_classes): #pass 0 because 0 is background
        pred_flat = y_pred[:, c].reshape(-1)
        true_flat = y_true[:, c].reshape(-1)
        intersection = (pred_flat * true_flat).sum()
       
        # with weight
        w = class_weights[c]/class_weights.sum()
        loss += w*(1 - ((2. * intersection + smooth) /
                         (pred_flat.sum() + true_flat.sum() + smooth)))
       
    return loss


def DSC(y_pred, y_true, ignore_background=True, smooth = 1.0):
    '''
    inputs:
        y_pred [n_classes, x, y, z] one-hot code
        y_true [n_classes, x, y, z] one-hot code
    '''
    smooth = 1.
    n_classes = y_pred.shape[0]
    dsc = []
    if ignore_background:
        for c in range(1, n_classes): #pass 0 because 0 is background
            pred_flat = y_pred[c, :].reshape(-1)
            true_flat = y_true[c, :].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
    else:
        for c in range(0, n_classes):
            pred_flat = y_pred[c, :].reshape(-1)
            true_flat = y_true[c, :].reshape(-1)
            intersection = (pred_flat * true_flat).sum()
            dsc.append(((2. * intersection + smooth) / (pred_flat.sum() + true_flat.sum() + smooth)))
            
        dsc = np.asarray(dsc)
        
    return dsc
