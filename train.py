#Import related libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import time
from dataset import train_x,train_y
from dataset import G1
from GRUGCN import GCN
import torch as th
import torch.nn as nn

def gcn_trainer(network, graph, input_data, label_data, training_times,
                optimizer, criterion, loss_list, dur_list):
    # loss_list = loss_list
    # network = network

    for epoch in range(training_times):
        t0 = time.time()
        network.train()
        out = network(graph, input_data)

        # criterion = criterion
        loss = criterion(out, label_data)

        # optimizer = optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_list.append(loss)
        dur_list.append(time.time() - t0)

        if (epoch + 1) % 70 == 0:  # epoch=70
            # acc = evaluate(net, g, features, labels, test_mask)
            print("Epoch {:04d} | MAE_Test_Loss {:.4f}".format(epoch + 1, loss.item()))

mymodel=GCN(6,12,1);GPGCN_Loss_list=[];GPGCN_Loss_list.clear();GPGCN_MAE_Dur_list=[]
for i in range(len(train_x)):
    print('Batch{:d}: '.format(i+1),end='')
    gcn_trainer(mymodel,G1,train_x[i],train_y[i],70,th.optim.Adam(mymodel.parameters(),lr=1e-3),nn.L1Loss(),GPGCN_Loss_list,GPGCN_MAE_Dur_list)
