#Import related libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch as th
import dgl
import pandas as pd
import numpy as np
from GRUGCN import GCN

A_city=pd.read_excel(r'D:\0_Shay\Practise\jupyter\CSJ\LJJZ.xlsx',index_col=[0])
A_city.fillna(value=0,inplace=True)

for i in A_city.columns:
    for j in A_city.index:
        if i==j:
            A_city[i][j]=1
A_city_array=np.array(A_city)
G1 = dgl.DGLGraph()

for i in range(len(A_city_array)):
    for j in range(len(A_city_array)):
        if A_city_array[i][j]==1:
            G1.add_edge(i, j)
        else:
            pass

#Input feature matrix
df = pd.read_excel(r'D:\0_Shay\Practise\jupyter\CSJ\GYH62.xlsx',index_col=[0])


def gen_lab(dataset, start, end, experience, future):
    data = []
    labels = []
    data_list = []
    labels_list = []

    real_start = start + experience

    for i in range(real_start, end):
        data.append(dataset.iloc[i - experience:i])
        labels.append(dataset.iloc[i:i + future])  # i:i+future

    for j in range(len(data)):
        data_tensor = th.Tensor(np.array(data[j].T))
        data_list.append(data_tensor)

    for k in range(len(labels)):
        labels_tensor = th.Tensor(np.array(labels[k].T))
        labels_list.append(labels_tensor)

    return th.stack(data_list), th.stack(labels_list)

#Divide the dataset according to 6:2:2
train_x,train_y = gen_lab(df,0,30,6,1)
valid_x,valid_y = gen_lab(df,30,40,6,1)
test_x,test_y = gen_lab(df,40,50,6,1)