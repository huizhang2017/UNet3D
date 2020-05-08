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

if __name__ == '__main__':
    
    torch.cuda.set_device(0) # assign which gpu will be used
    
    train_list = './train_list.csv'
    val_list = './val_list.csv'
    
    num_classes = 3
    num_channels = 1
    num_epochs = 1
    num_workers = 4
    train_batch_size = 20
    val_batch_size = 1
    num_batches_print = 20
    model_name = 'unet3d_test'
    
    # set plotter
    global plotter
    plotter = utils.VisdomLinePlotter(env_name=model_name)
    
    # mkdir 'models'
    if not os.path.exists('./models'):
        os.mkdir('./models')
    
    # set dataset
    training_dataset = CBCT_Dataset(train_list)
    val_dataset = CBCT_Dataset(val_list)
    
    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    unet3d = UNet3D(in_channels=num_channels, out_channels=num_classes ).to(device, dtype=torch.float)
    opt = optim.Adam(unet3d.parameters(), lr=0.001)
    
    losses, metrics = [], []
    val_losses, val_metrics = [], []
    
    best_val_dsc = 0.0
    
    for epoch in range(num_epochs):

        # training
        unet3d.train()
        running_loss = 0.0
        running_metric = 0.0
        loss_epoch = 0.0
        metric_epoch = 0.0
        class_weights = torch.Tensor([0.05, 1.0, 2.0]).to(device, dtype=torch.float).to(device, dtype=torch.float)
        
        for i_batch, batched_sample in enumerate(train_loader):

            # send mini-batch to device            
            inputs, labels = batched_sample['image'].to(device, dtype=torch.float), batched_sample['label'].to(device, dtype=torch.long)
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :, :, :], num_classes=num_classes)
            one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3)
            
            # zero the parameter gradients
            opt.zero_grad()
            
            # forward + backward + optimize
            outputs = unet3d(inputs)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights) 
            metric = weighting_DSC(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()
            
            # print statistics
            running_loss += loss.item()
            running_metric += metric.item()
            loss_epoch += loss.item()
            metric_epoch += metric.item()
            if i_batch % num_batches_print == num_batches_print-1:  # print every 200 mini-batches
                print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_print, running_metric/num_batches_print))
                running_loss = 0.0
                running_metric = 0.0
        
        # record training progress and plot
        losses.append(loss_epoch/len(train_loader))
        metrics.append(metric_epoch/len(train_loader))
        
        # save the checkpoint
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': unet3d.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'loss': loss,
                    'metric': metric}, './models/checkpoint.tar')
        
        #reset
        loss_epoch = 0.0
        metric_epoch = 0.0
        torch.cuda.empty_cache()
                
        # validation
        unet3d.eval()
        with torch.no_grad():
            val_loss_epoch = 0.0
            val_metric_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):
                
                # send mini-batch to device
                val_inputs, val_labels = batched_val_sample['image'].to(device, dtype=torch.float), batched_val_sample['label'].to(device, dtype=torch.long)
                one_hot_labels = nn.functional.one_hot(val_labels[:, 0, :, :, :], num_classes=num_classes)
                one_hot_labels = one_hot_labels.permute(0, 4, 1, 2, 3)
                
                val_outputs = unet3d(val_inputs)
                val_loss = Generalized_Dice_Loss(val_outputs, one_hot_labels, class_weights)
                val_metric = weighting_DSC(val_outputs, one_hot_labels, class_weights)
                
                val_loss_epoch += val_loss.item()
                val_metric_epoch += val_metric.item()
                
                print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), val_loss, val_metric))
            
            val_losses.append(val_loss_epoch/len(val_loader))
            val_metrics.append(val_metric_epoch/len(val_loader))
            val_loss_epoch = 0.0
            val_metric_epoch = 0.0
            
            # save the best model
            if best_val_dsc < val_metric_epoch/len(val_loader):
                best_val_dsc = val_metric_epoch/len(val_loader)
                torch.save(unet3d.state_dict(), './models/{0}_best.pt'.format(model_name))
            
            # output current status
            print('*****\n[Epoch: {0}/{1}, loss: {2}, dsc: {3}\n     val_loss: {4}, val_dsc: {5}\n*****'.format(epoch+1, num_epochs, losses[-1], metrics[-1], val_losses[-1], val_metrics[-1]))
            plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
            plotter.plot('DSC', 'train', 'DSC', epoch+1, metrics[-1])
            plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
            plotter.plot('DSC', 'val', 'DSC', epoch+1, val_metrics[-1])
            torch.cuda.empty_cache()
            
    # save all losses and metrics data
    pd_dict = {'loss': losses, 'DSC': metrics, 'val_loss': val_losses, 'val_DSC': val_metrics}
    stat = pd.DataFrame(pd_dict)
    stat.to_csv('losses_metrics_vs_epoch.csv')
            