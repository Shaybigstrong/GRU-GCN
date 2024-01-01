#Import related libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch as th
import pandas as pd
import numpy as np



# The following is a multi-step prediction under recursive strategy
def gen_labpre(dataset, start, end, experience, future):
    data = []
    labels = []
    data_list = []
    labels_list = []

    real_start = start + experience

    for i in range(real_start, end + 1):
        data.append(dataset.iloc[i - experience:i])
        labels.append(dataset.iloc[i:i + future])  # i:i+future

    for j in range(len(data)):
        data_tensor = th.Tensor(np.array(data[j].T))
        data_list.append(data_tensor)

    for k in range(len(labels)):
        labels_tensor = th.Tensor(np.array(labels[k].T))
        labels_list.append(labels_tensor)

    return th.stack(data_list), th.stack(labels_list)

CSJmodel = th.load(r'D:\0_Shay\Practise\jupyter\trained\aftertained\CSJ.pth')
testlist = pd.read_excel(r'D:\0_Shay\Practise\jupyter\trained\prediction4.xlsx', index_col=[0])

for i in range(4):
    test_xpre, test_ypre = gen_labpre(testlist, 0, 6 + i, 6, 0)
    predict1 = CSJmodel(G1, test_xpre[-1])
    predict1
    out_list = predict1.tolist()
    out_list = np.transpose(out_list).tolist()
    testlist = testlist.append(out_list)