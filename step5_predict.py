import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataset import *
from UNet3D import *
from loss import *
import utils
import pandas as pd
import itk

def get_stride(image_width, kernel_size):
    '''
    return proper stride that can slide all images with min number of steps (min number of patches)
           by the given image and kernel sizes
    '''
    n = image_width//kernel_size + 1
    stride = (image_width - kernel_size) // (n - 1)
    return stride

if __name__ == '__main__':
    
    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
      
    model_path = './models/'
    model_name = 'checkpoint.tar'
    
    image_path = 'D:/Downloads/Alveolar_Cleft/GroundTruth/flip_Res0p4_smoothed/'
    test_list = [27]
    test_image_filename = 'NORMAL0{0}_cbq-n3-7.hdr'
    test_label_filename = 'NORMAL0{0}-ls-corrected-ordered-smoothed.nrrd'
    
    num_classes = 3
    num_channels = 1
    patch_size = np.array([128, 128, 128])
          
    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D(in_channels=num_channels, out_channels=num_classes).to(device, dtype=torch.float)
    
    # load trained model
    checkpoint = torch.load(model_path+model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    
    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    # eval model first to check gpu memory    
    print('Testing')
    model.eval()
    with torch.no_grad():
        for i_sample in test_list:
            
            # read image and label (annotation)
            itk_image = itk.imread(image_path+test_image_filename.format(i_sample))
            itk_annotation = itk.imread(image_path+test_label_filename.format(i_sample))
            np_image = itk.array_from_image(itk_image)
            np_label = itk.array_from_image(itk_annotation)
            predict_label = np.zeros(np_label.shape)
        
            # normalized
            np_image = (np_image - np_image.mean())/ np_image.std()
            np_image = np_image.reshape([1, np_image.shape[0], np_image.shape[1], np_image.shape[2]])
            
            # numpy -> torch.tensor
            tensor_image = torch.from_numpy(np_image).to(device, dtype=torch.float)
            
            # get stirde, will use when fold
            stride_1 = get_stride(np_image.shape[1], patch_size[0])
            stride_2 = get_stride(np_image.shape[2], patch_size[1])
            stride_3 = get_stride(np_image.shape[3], patch_size[2])
            
            # get num of strides
            n1 = (np_image.shape[1]-patch_size[0])/stride_1 + 1
            n2 = (np_image.shape[2]-patch_size[1])/stride_2 + 1
            n3 = (np_image.shape[3]-patch_size[2])/stride_3 + 1
            
            # create patches covering the entire image
            image_patches = tensor_image.unfold(1, patch_size[0], get_stride(np_image.shape[1], patch_size[0])).unfold(2, patch_size[1], get_stride(np_image.shape[2], patch_size[1])).unfold(3, patch_size[2], get_stride(np_image.shape[3], patch_size[2]))
            image_patches = image_patches.reshape(-1, 1, patch_size[0], patch_size[1], patch_size[2])
            patch_image = image_patches
            
            # send mini-batch to device
#            val_inputs, val_labels = batched_val_sample['image'].to(device, dtype=torch.float), batched_val_sample['label'].to(device, dtype=torch.long)
#            one_hot_labels = nn.functional.one_hot(val_labels[:, 0, :, :, :], num_classes=num_classes)
#            one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3)
           
            patch_output = np.zeros(patch_image.shape)
            for i_patch in range(patch_image.shape[0]):
                tensor_prob_output = model(patch_image[i_patch, :, :, :, :,].view(-1, num_channels, patch_size[0], patch_size[1], patch_size[2])).detach()
                patch_prob_output = tensor_prob_output.cpu().numpy()
                
                for i_label in range(num_classes):
                    patch_output[i_patch, np.argmax(patch_prob_output[:], axis=1)==i_label] = i_label
                    
            # squeeze
            patch_output = np.squeeze(patch_output, axis=1)
            
            # self fold function: combine patch results
            i_patch = 0
            for k in range(int(n1)):
                for j in range(int(n2)):
                    for i in range(int(n3)):
                        predict_label[0+int(stride_1*k):patch_size[0]+int(stride_1*k), 0+int(stride_2*j):patch_size[1]+int(stride_2*j), 0+int(stride_3*i):patch_size[2]+int(stride_3*i)] = patch_output[i_patch, :]
                        i_patch += 1
                        
            # output result
            itk_predict_label = itk.image_view_from_array(predict_label)
            itk.imwrite(itk_predict_label, 'test_result.nrrd')
            
    
            