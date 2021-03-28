import numpy as np
import os
import cv2
import glob
import pydicom
import re
from natsort import natsorted

def createFile(fileName):
    if os.path.exists(fileName): # if the path already exists do nothing
        pass
    else:
        try:
            os.makedirs(fileName)       #創建資料夾
        except IOError:
            print(fileName + '<- This name causes an error')


def resize_img(datalabel):
    size = 512
    y = np.zeros((len(datalabel),size,size),dtype=np.uint8)
    for i in range(len(datalabel)):
        y[i] = cv2.resize(datalabel[i], (size, size), interpolation=cv2.INTER_NEAREST)
        y[i] = cv2.medianBlur(y[i], 3)
    return y



class predict_to_image(object):
    def __init__(self,Pred,pred_Name,AI_pred_path):
		# input CT or RT data
        self.Pred=Pred
        self.pred_Name=pred_Name
        self.save_Path=AI_pred_path
    
    def __call__(self):

        new_file_name = []

        dcm_path = './'+ self.pred_Name 
        file_name = os.listdir(dcm_path)
        file_name = natsorted(file_name,reverse=False)
        file_name.pop()
    
        for file in file_name :
            dcm_file = dcm_path + '/' + file 
            ds = pydicom.dcmread(dcm_file)
            file_name = ds.SOPInstanceUID

            new_file_name.append(file_name)
            #print(new_file_name)

        createFile(self.save_Path)
      
        
        pred = np.load(self.Pred)
        length=len(pred)
        pred = pred[:length]
        pred = np.argmax(pred, axis=-1)
        pred = resize_img(pred)
        print(pred.shape)

        for j in range(len(pred)):
            cv2.imwrite(self.save_Path + '/' + new_file_name[j] +'.bmp' , pred[j]*255)
