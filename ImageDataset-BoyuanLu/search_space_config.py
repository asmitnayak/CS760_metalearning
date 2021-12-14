#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 20:03:26 2021

@author: lby
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F 
import pandas as pd

ARCHITECTURES = ['SQU','ASC','DES','SYM','ENC']
ACTIVATION_FS = ['ReLu','Selu']
DROPOUT = [0, 0.5]
HIDDEN_LAYERS = [3, 5, 7, 1]
NEURONS = [4, 8, 16, 32]
def get_units(idx,neurons,architecture,layers=None):
    assert architecture in ARCHITECTURES
    if architecture == 'SQU':
        return neurons
    elif architecture == 'ASC':
        return (2**idx)*neurons
    elif architecture == 'DES':
        return (2**(layers-1-idx))*neurons    
    elif architecture=='SYM':
        assert (layers is not None and layers > 2)
        if layers%2==1:
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1
            return neurons*2**(int(layers/2) - abs(int(layers/2)-idx))
        
    elif architecture=='ENC':
        assert (layers is not None and layers > 2)
        if layers%2==0:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx -1            
            return neurons*2**(int(layers/2)-1 -idx)
        else:
            x = int(layers/2)
            idx  = idx if idx < x else 2*x-idx
            return neurons*2**(int(layers/2) -idx)
        
def Data_Loader(batchsize):
    transform = transforms.ToTensor()
    
    train_data = datasets.CIFAR10(root='../Data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='../Data', train=False, download=True, transform=transform)
    
    torch.manual_seed(101)  # for consistent results
    
    train_loader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    
    test_loader = DataLoader(test_data, batch_size=batchsize, shuffle=False)    
    
    return train_loader, test_loader
    
    
class nn_config():
    def __init__(self, architecture, activation_f, dropout, hidden_layers, neurons):
        self.architecture = architecture
        self.activation_f = activation_f
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        
        self.get_neurons()
        
        
    def get_neurons(self):
        neurons_units = []
        for i in range(self.hidden_layers):
            neurons = self.neurons
            units = get_units((i), neurons, self.architecture, self.hidden_layers)
            neurons_units.append(units)
        
        self.hidden_neurons_list = neurons_units
        
class NetworkModel(nn.Module,nn_config):
    def __init__(self, architecture, activation_f, dropout, hidden_layers, neurons):
        nn.Module.__init__(self)
        nn_config.__init__(self, architecture, activation_f, dropout, hidden_layers, neurons)

        # mnist dataset dimension
        n_in = 32*32*3
        n_out = 10
        # create a layer list
        
        layerlist = []
        for i in self.hidden_neurons_list:
            layerlist.append(nn.Linear(n_in,i))
            if self.activation_f == 'ReLu':
                layerlist.append(nn.ReLU(inplace=True))
            else:
                layerlist.append(nn.SELU(inplace=True))
            layerlist.append(nn.Dropout(self.dropout))
            n_in = i 
        layerlist.append(nn.Linear(self.hidden_neurons_list[-1],n_out))
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x):
        return F.log_softmax(self.layers(x),dim=1)


search_space = pd.read_csv('mnist_search_table.txt', index_col=None,header=None)
for instance in range(search_space.shape[0]):
    
    # dropout
    if search_space.iloc[instance,0] == 0:
        dropout = DROPOUT[0]
    else:
        dropout = DROPOUT[1]
        
    # hidden layer
    if search_space.iloc[instance,1] == 1/3:
        hidden_layers = HIDDEN_LAYERS[0]
    else:
        hidden_layers = HIDDEN_LAYERS[1]
        
    # neurons
    if search_space.iloc[instance,2] == 0:
        neurons = NEURONS[0]
    elif search_space.iloc[instance,2] < 1.1/7 and search_space.iloc[instance,2] > 0.9/7:
        neurons = NEURONS[1]
    elif search_space.iloc[instance,2] < 3.1/7 and search_space.iloc[instance,2] > 2.9/7:
        neurons = NEURONS[2]
    else:
        neurons = NEURONS[3]
    
    # activation function
    if search_space.iloc[instance,3] == 1:
        activation_f = ACTIVATION_FS[0]
    elif search_space.iloc[instance,4] == 1:
        activation_f = ACTIVATION_FS[1]
    else:
        print('activation function error')
        break
    
    # architectures
    if search_space.iloc[instance,5] == 1:
        architecture = ARCHITECTURES[0]
    elif search_space.iloc[instance,6] == 1:
        architecture = ARCHITECTURES[1]
    elif search_space.iloc[instance,7] == 1:
        architecture = ARCHITECTURES[2]
    elif search_space.iloc[instance,8] == 1:
        architecture = ARCHITECTURES[3]
    elif search_space.iloc[instance,9] == 1:
        architecture = ARCHITECTURES[4]
    else:
        print('architecture error')
        break
    
    
    model = NetworkModel(architecture, activation_f, dropout, hidden_layers, neurons)
    
    
    
    # stay with the dataset2vec code
    epochs = 10000
    batch_size = 16 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_loader, test_loader = Data_Loader(batch_size)
    
    for i in range(epochs):
        for b, (X_train, y_train) in enumerate(train_loader):
            
            y_pred = model(X_train.view(batch_size,-1))
            loss = criterion(y_pred,y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if b>0:
                break
    tst_corr = 0
    with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
    
                # Apply the model
                y_val = model(X_test.view(batch_size, -1))  # Here we flatten X_test
    
                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1] 
                batch_corr = (predicted == y_test).sum()
                tst_corr += batch_corr
            acc = tst_corr/10000    
            search_space.iloc[instance,10] = acc.item()
    print(f'[{instance}/255] dropout:{dropout} act_f:{activation_f} layers:{hidden_layers} neurons:{neurons} archi:{architecture} acc:{acc}')
search_space.to_csv('cifar10_mnist_search_table.txt' ,header=None)