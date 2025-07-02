import warnings
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import os
warnings.filterwarnings("ignore")

# Switcher(0)_Caltech101_7
def Caltech101_7() :
    # mdata1 = sio.loadmat('C:/Users/asus/Desktop/Dataset/C_1_3.mat')
    # mdata2 = sio.loadmat('C:/Users/asus/Desktop/Dataset/C_4_6.mat')
    # mLabels = sio.loadmat('C:/Users/asus/Desktop/Dataset/C_label.mat')
    # mdata1 = sio.loadmat('C:/Users/asus/Desktop/Dataset/C_1_3.mat')
    # mdata2 = sio.loadmat('C:/Users/asus/Desktop/Dataset/C_4_6.mat')
    # mLabels = sio.loadmat('C:/Users/asus/Desktop/Dataset/C_label.mat')
    mdata = sio.loadmat(r'C:/Users/asus/Desktop/桌面勿动/资料/Dataset/Caltech101-7.mat')
    S_list = []
    S_list.append(np.array(mdata['X'][0][0]))
    S_list.append(np.array(mdata['X'][0][1]))
    S_list.append(np.array(mdata['X'][0][2]))
    S_list.append(np.array(mdata['X'][0][3]))
    S_list.append(np.array(mdata['X'][0][4]))
    S_list.append(np.array(mdata['X'][0][5]))
    gnd = np.squeeze(mdata['Y'])
    return S_list, gnd

# Switcher(1)_3sources
def sources():
    # data = sio.loadmat('C:/Users/asus/Desktop/Dataset/3sources.mat')
    data = sio.loadmat(r'C:/Users/asus/Desktop/桌面勿动/资料/Dataset/3sources.mat')
    S_list = []
    S_list.append(np.array(data['data'][0][0]))
    S_list.append(np.array(data['data'][0][1]))
    S_list.append(np.array(data['data'][0][2]))
    gnd = np.squeeze(np.array(data['truelabel'][0][0])).flatten()
    return S_list, gnd

# Switcher(2)_MSRC_v1
def MSRC():
    # data = sio.loadmat('C:/Users/asus/Desktop/Dataset/MSRCv1.mat')
    data = sio.loadmat(r'C:/Users/asus/Desktop/桌面勿动/资料/Dataset/MSRC.mat')
    S_list = []
    S_list.append(np.array(data['X'][0][0]))
    S_list.append(np.array(data['X'][0][1]))
    S_list.append(np.array(data['X'][0][2]))
    S_list.append(np.array(data['X'][0][3]))
    S_list.append(np.array(data['X'][0][4]))
    gnd = np.squeeze(data['Y'])
    return S_list, gnd

# Switcher(3)_ORL
def Orl():
    # data = sio.loadmat('C:/Users/asus/Desktop/Dataset/ORL.mat')
    data = sio.loadmat(r'C:/Users/asus/Desktop/桌面勿动/资料/Dataset/ORL.mat')
    S_list = []
    S_list.append(np.array(data['X'][0][0]))
    S_list.append(np.array(data['X'][0][1]))
    S_list.append(np.array(data['X'][0][2]))
    S_list.append(np.array(data['X'][0][3]))
    gnd = np.squeeze(data['Y'])
    return S_list, gnd

# Switcher(4)_UCI
def BBCSport():
    # mat_data = sio.loadmat('C:/Users/asus/Desktop/Dataset/BBCSport.mat')
    mat_data = sio.loadmat(r'C:/Users/asus/Desktop/桌面勿动/资料/Dataset/BBCSport.mat')
    data = mat_data['data']
    truelabel = mat_data['truelabel']
    S_list = [data[0, 0].T, data[0, 1].T]
    gnd = truelabel[0, 0].flatten()
    return S_list, gnd


Switcher = {
    0: sources,
    1: Caltech101_7,
    2: MSRC,
    3: Orl,
    4: BBCSport
}