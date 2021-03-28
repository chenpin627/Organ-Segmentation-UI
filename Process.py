import numpy as np
import os
import cv2
import keras
import time as t
import datetime
from numpy.lib.format import open_memmap


global Multip_Organ
global height
global width
global time
global n_labels

height = 512
width = 512
time = 3
n_labels = 2
Multip_Organ = ['Liver', 'Esophagus', 'Heart', 'Lung', 'Esophagus','Kidney']
#Multip_Organ = ['Liver']
#Multip_Organ = ['Stomach']
#Multip_Organ = ['Esophagus']
#Multip_Organ = ['Heart']
#Multip_Organ = ['Lung']

def resize_img_test(dataArray):
    size = 256
    x  = np.zeros((len(dataArray),3 ,size,size), dtype=np.uint8)
    for i in range(len(dataArray)):
        x[i,0] = cv2.resize(dataArray[i,0], (size, size), interpolation=cv2.INTER_LINEAR)
        x[i,1] = cv2.resize(dataArray[i,1], (size, size), interpolation=cv2.INTER_LINEAR)
        x[i,2] = cv2.resize(dataArray[i,2], (size, size), interpolation=cv2.INTER_LINEAR)
     
    return x


def CLAHE(npy):
    CT_height, CT_width, CTslice  = npy.shape
    for i in range(CTslice):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        npy[:,:,i] = clahe.apply(npy[:,:,i])

    return npy 



def ProcessTimeData(CT, time=3, data=None):
    if data=='x':
        CT_height, CT_width, CTslice = CT.shape
    elif data=='y':
        CTslice, CT_width, CT_height = CT.shape

    padNum = int( (time-1) / 2 )

    if data =='x':
        padTop = CT[:,:,0]
        padBtm = CT[:,:,-1]
        for i in range(padNum):
            CT = np.insert(CT, 0, values=padTop, axis=-1)
            CT = np.insert(CT, -1, values=padBtm, axis=-1)
        
        CT = CT[np.newaxis,:]    
        concat = np.empty((CTslice,time,CT_height,CT_width), dtype=CT.dtype)

        for j in np.arange(padNum, CTslice+padNum):
            for d in range(time):
                concat[j-padNum,d] = CT[:,:,:,j-padNum + d]


    elif data=='y':
        padTop = CT[0,:,:]
        padBtm = CT[-1:,:]
        concat = np.empty((CTslice,time,CT_height,CT_width), dtype=CT.dtype)
        for i in range(padNum):
            CT = np.insert(CT, 0, values=padTop, axis=0)
            CT = np.insert(CT, -1, values=padBtm, axis=0)
        for j in np.arange(padNum, CTslice+padNum):
             for d in range(time):
                concat[j-padNum,d] = CT[j-padNum + d]

    print(concat.shape)
# =============================================================================
#     concat = np.delete(concat, 0, axis=0)
#     concat = np.delete(concat, -1, axis=0)
# =============================================================================
    return concat    

def TimeDataGroundTruth(path):
    lis = []
    # kkk = 0
    paths = sorted(os.listdir(path), reverse=False)

    LabelArray = np.empty((1,512,512), dtype=np.uint8)
    for file in paths:

        image = cv2.imread(os.path.join(path, file), 0)

        image[np.where(image<=0)] = 0
        image[np.where(image!=0)] = 1
        image = np.expand_dims(image, axis=0)

        # image = image.reshape((1,1,image.shape[0],image.shape[1]))
        LabelArray = np.append(LabelArray, image, axis=0)
    
    LabelArray = np.delete(LabelArray,0,0)   
    return LabelArray


def get_Sensor_data(img, label, Organ_list, time):
    path = os.listdir(img)
    path = sorted(path, reverse=False)
    print(path)
    index = []

    Train_x = np.empty((1,time,512,512), dtype=np.uint8)
    Label_y = np.empty((1,time,512,512), dtype=np.uint8)

    for i in path:
        folder = img + '/' + i + '/' + 'DicomArray.npy'
        img_array = np.load(folder)

        img_array = CLAHE(img_array)
        img_array = ProcessTimeData(img_array, time,'x')

        Organ = Organ_list 
        folder = label + '/' + i + '/' + 'GroundTruth' + '/' + Organ
        label_array = TimeDataGroundTruth(folder)
        label_array = ProcessTimeData(label_array, time, 'y')

        if (len(img_array)==label_array.shape[0]):
            print("fold: " + str(i) + "---ok---")
            Train_x = np.concatenate((Train_x,img_array), axis=0)
            Label_y = np.concatenate((Label_y,label_array), axis=0)
            print(len(img_array))
            index.append(len(img_array))
            # Train_x = np.append(Train_x, img_array, axis=0)
            # Label_y = np.append(Label_y, label_array, axis=0)
        else:
            print("fold: " + str(i) + "---Number of x and y is not match---")

    Train_x = np.delete(Train_x, 0, axis=0)
    Label_y = np.delete(Label_y, 0, axis=0)

    return Train_x, Label_y, index        



def get_Sensor_data_test(img , time):
    path = os.listdir(img)
    path = sorted(path, reverse=False)
    print(path)
    index = []

    Train_x = np.empty((1,time,512,512), dtype=np.uint8)


    folder = img + 'DicomArray.npy'
    img_array = np.load(folder)

    img_array = CLAHE(img_array)
    img_array = ProcessTimeData(img_array, time,'x')
    print("fold: " + str(img) + "---ok---")
    Train_x = np.concatenate((Train_x,img_array), axis=0)
    print(len(img_array))
    index.append(len(img_array))


# =============================================================================
#     for i in path:
#         folder = img + '/' + i + '/' + 'DicomArray.npy'
#         img_array = np.load(folder)
# 
#         img_array = CLAHE(img_array)
#         img_array = ProcessTimeData(img_array, time,'x')
#         print("fold: " + str(i) + "---ok---")
#         Train_x = np.concatenate((Train_x,img_array), axis=0)
#         print(len(img_array))
#         index.append(len(img_array))
# =============================================================================

    Train_x = np.delete(Train_x, 0, axis=0)

    return Train_x, index        
        
class processtrain(object):
    def __init__(self,savepathCT,savepathRT):
		# input CT or RT data
        self.savepathCT=savepathCT
        self.savepathRT=savepathRT
    
    
    def __call__(self):
        ISOTIMEFORMAT = '%Y%m%d'
        date=datetime.date.today().strftime(ISOTIMEFORMAT)
        
        for i in range(len(Multip_Organ)):
            test_img, test_label, index = get_Sensor_data(self.savepathCT, self.savepathRT, Multip_Organ[i], 3)
            test_img, test_label = resize_img(test_img, test_label)
            test_img = np.expand_dims(test_img, axis=-1)
            print(test_img.shape)
            print(test_img.dtype)
            print(test_label.shape)
            print(test_label.dtype)
    
            save_name= 'train_data_'+ date +'_' + Multip_Organ[i] + '.npz'
            #save_name= 'train_data_20200601' + Multip_Organ[i] + '_.npz'
            #save_name= 'test_data_10P_liver_50_175_' + Multip_Organ[i] + '_.npz'
            np.savez_compressed(save_name, X = test_img, Y = test_label, Index=index)


class processtest(object):
    def __init__(self,savepathCT,process_save_name):
		# input CT or RT data
        self.savepathCT=savepathCT
        self.save_name=process_save_name
        
    def __call__(self):
        #ISOTIMEFORMAT = '%Y%m%d'
        #date=datetime.date.today().strftime(ISOTIMEFORMAT)
        
        test_img, index = get_Sensor_data_test(self.savepathCT, 3)
        test_img= resize_img_test(test_img)
        test_img = np.expand_dims(test_img, axis=-1)
        print(test_img.shape)
        print(test_img.dtype)

        #DCM_FolderName=self.DCM_FolderName
        
        #save_name= './Data/' + date + '_'+ DCM_FolderName + '.npz'        
        np.savez_compressed(self.save_name, X = test_img, Index=index)

        




