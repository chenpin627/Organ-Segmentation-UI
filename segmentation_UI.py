from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox
from PyQt5.QtWidgets import QFileDialog
import sys
import os
import datetime

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import glob
import numpy as np
from tqdm import tqdm
from getArray import *

from Process import processtrain
from Process import processtest

from deeplearning import *
from predict_to_image import *
from SegmentiontoImageData import *
from ImagetoRT import *
import importlib
from natsort import natsorted

importlib.reload(sys)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(860, 618)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setReadOnly(True)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 1, 2, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 1, 1, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout.addWidget(self.pushButton_2, 1, 3, 1, 1)
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 0, 3, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.lineEdit.setFont(font)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout.addWidget(self.lineEdit, 0, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 2, 1, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setReadOnly(True)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 2, 2, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_3.setObjectName("checkBox_3")
        self.gridLayout_2.addWidget(self.checkBox_3, 0, 2, 1, 1)
        self.checkBox_2 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_2.setObjectName("checkBox_2")
        self.gridLayout_2.addWidget(self.checkBox_2, 0, 1, 1, 1)
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_2.addWidget(self.checkBox, 0, 0, 1, 1)
        self.checkBox_4 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_4.setObjectName("checkBox_4")
        self.gridLayout_2.addWidget(self.checkBox_4, 0, 3, 1, 1)
        self.checkBox_5 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_5.setObjectName("checkBox_5")
        self.gridLayout_2.addWidget(self.checkBox_5, 0, 4, 1, 1)
        self.checkBox_6 = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox_6.setObjectName("checkBox_6")
        self.gridLayout_2.addWidget(self.checkBox_6, 0, 5, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_2)
        self.verticalLayout.addWidget(self.groupBox)
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.textBrowser.setFont(font)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout.addWidget(self.textBrowser)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
        #button點擊事件
        self.pushButton.clicked.connect(self.dicom_openfile)
        self.pushButton_2.clicked.connect(self.dicomRT_openfile)
        self.pushButton_3.clicked.connect(self.Prediction_Start)




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Organ Segmentation"))
        self.label_2.setText(_translate("MainWindow", "DicomRT Path："))
        self.pushButton_2.setText(_translate("MainWindow", "Load DicomRT"))
        self.pushButton.setText(_translate("MainWindow", "Load Data"))
        self.label.setText(_translate("MainWindow", "Dicom Data Path："))
        self.label_3.setText(_translate("MainWindow", "Save DicomRT Path："))
        self.groupBox.setTitle(_translate("MainWindow", "Organ"))
        self.checkBox_3.setText(_translate("MainWindow", "Stomach"))
        self.checkBox_2.setText(_translate("MainWindow", "Liver"))
        self.checkBox.setText(_translate("MainWindow", "Lung"))
        self.checkBox_4.setText(_translate("MainWindow", "Esophagus"))
        self.checkBox_5.setText(_translate("MainWindow", "Heart"))
        self.checkBox_6.setText(_translate("MainWindow", "Kidney"))
        self.pushButton_3.setText(_translate("MainWindow", "Prediction Start"))

    def printf(self, mes):
        self.textBrowser.append(mes)  
        self.cursot = self.textBrowser.textCursor()
        self.textBrowser.moveCursor(self.cursot.End)
        QtWidgets.QApplication.processEvents() 
        
    
    #RTtoImage           
    def dicom_openfile(self): 
        self.lineEdit.clear()
        self.lineEdit_2.clear()
        self.lineEdit_3.clear()
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(False)
        
        DCM_Folders = QtWidgets.QFileDialog.getExistingDirectory(None,"選取資料夾","./")  
        
        self.lineEdit.setText(DCM_Folders)
        ui.printf('Load_Dicom_Finish') 
        self.pushButton.setEnabled(False)
        self.pushButton_2.setEnabled(True)  


    def dicomRT_openfile(self): 
        dicomRT_file ,RT_filetype= QtWidgets.QFileDialog.getOpenFileName(None,"選取檔案","./","Files (*.dcm)")
        self.lineEdit_2.setText(dicomRT_file)
        ui.printf('Load_DicomRT_Finish')   
        self.pushButton_2.setEnabled(False)
        self.pushButton_3.setEnabled(True)  
        
        
    def Prediction_Start(self): 
        
        ISOTIMEFORMAT = '%Y%m%d'
        date=datetime.date.today().strftime(ISOTIMEFORMAT)
        
        DCM_Folders = self.lineEdit.text()        
        DCM_FolderName = DCM_Folders.split("/")[-1]    #擷取路徑中最後一層資料夾名字 
        
        DCM_Folder_ID = DCM_Folders + '/'
        list_FileCT = glob.glob(DCM_Folder_ID + 'CT*.dcm')
        list_FileCT = natsorted(list_FileCT,reverse=False)
        SaveCT_Path =  './Data/'+ date + '_'+ DCM_FolderName + '/'
        getArray(list_FileCT, DCM_Folder_ID, SaveCT_Path)()

        process_save_name= './Data/'+date + '_'+ DCM_FolderName +'/'+ date + '_'+ DCM_FolderName + '.npz'    
        processtest(SaveCT_Path,process_save_name)()   
        ui.printf('Preprocess_Finish')


        #Deep Learning 
        check_organ=[]   
        TestDataName=process_save_name
        pred_FolderName = TestDataName.split("/")[-1]    #擷取路徑中最後一層資料夾名字
        pred_FolderName = pred_FolderName.split("_")[-1]    #擷取路徑中最後一層資料夾名字
        pred_FolderName,Type = pred_FolderName.split(".")    #擷取路徑中最後一層資料夾名字

    
        if self.checkBox.isChecked():
            pred_organ=self.checkBox.text()
            ui.printf(pred_organ + '_Prediction_Start')  
            WeightPath='./weight/Fusion_lung_weights.h5'
            Save_Max_ModelName_test = './Data/'+ date + '_'+ pred_FolderName +'/' + pred_FolderName + '_' +pred_organ + '_' +"Max_model.h5"
            MaxSavePred = './Data/'+date + '_'+ pred_FolderName +'/' + pred_FolderName + '_'+ pred_organ + '_' +"Max_pred.npy"
            Prediction(TestDataName,WeightPath,Save_Max_ModelName_test,MaxSavePred)()   
            AI_pred_path= './Data/'+ date + '_'+ pred_FolderName + '/AI/' + 'AI' + pred_organ 
            predict_to_image(MaxSavePred,pred_FolderName,AI_pred_path)() 
            ui.printf(pred_organ + '_Prediction_Finish')  
                 
        if self.checkBox_2.isChecked():
            pred_organ=self.checkBox_2.text()
            ui.printf(pred_organ + '_Prediction_Start')  
            WeightPath='./weight/Fusion_liver_weights.h5'
            Save_Max_ModelName_test = './Data/'+ date + '_'+ pred_FolderName +'/' + pred_FolderName + '_' +pred_organ + '_' +"Max_model.h5"
            MaxSavePred = './Data/'+date + '_'+ pred_FolderName +'/' + pred_FolderName + '_'+ pred_organ + '_' +"Max_pred.npy"
            Prediction(TestDataName,WeightPath,Save_Max_ModelName_test,MaxSavePred)()   
            AI_pred_path= './Data/'+ date + '_'+ pred_FolderName + '/AI/' + 'AI' + pred_organ 
            predict_to_image(MaxSavePred,pred_FolderName,AI_pred_path)() 
            ui.printf(pred_organ + '_Prediction_Finish')          
            
        if self.checkBox_3.isChecked():
            pred_organ=self.checkBox_3.text()
            ui.printf(pred_organ + '_Prediction_Start')  
            WeightPath='./weight/Fusion_stomach_weights.h5'
            Save_Max_ModelName_test = './Data/'+ date + '_'+ pred_FolderName +'/' + pred_FolderName + '_' +pred_organ + '_' +"Max_model.h5"
            MaxSavePred = './Data/'+date + '_'+ pred_FolderName +'/' + pred_FolderName + '_'+ pred_organ + '_' +"Max_pred.npy"
            Prediction(TestDataName,WeightPath,Save_Max_ModelName_test,MaxSavePred)()     
            AI_pred_path= './Data/'+ date + '_'+ pred_FolderName + '/AI/' + 'AI' + pred_organ 
            predict_to_image(MaxSavePred,pred_FolderName,AI_pred_path)() 
            ui.printf(pred_organ + '_Prediction_Finish')  
        
        if self.checkBox_4.isChecked():
            pred_organ=self.checkBox_4.text()
            ui.printf(pred_organ + '_Prediction_Start')    
            WeightPath='./weight/Fusion_Esophagus_weights.h5'
            Save_Max_ModelName_test = './Data/'+ date + '_'+ pred_FolderName +'/' + pred_FolderName + '_' +pred_organ + '_' +"Max_model.h5"
            MaxSavePred = './Data/'+date + '_'+ pred_FolderName +'/' + pred_FolderName + '_'+ pred_organ + '_' +"Max_pred.npy"
            Prediction(TestDataName,WeightPath,Save_Max_ModelName_test,MaxSavePred)()     
            AI_pred_path= './Data/'+ date + '_'+ pred_FolderName + '/AI/' + 'AI' + pred_organ 
            predict_to_image(MaxSavePred,pred_FolderName,AI_pred_path)() 
            ui.printf(pred_organ + '_Prediction_Finish')  
            
        if self.checkBox_5.isChecked():
            pred_organ=self.checkBox_5.text()
            ui.printf(pred_organ + '_Prediction_Start')  
            WeightPath='./weight/Fusion_Heart_weights.h5'
            Save_Max_ModelName_test = './Data/'+ date + '_'+ pred_FolderName +'/' + pred_FolderName + '_' +pred_organ + '_' +"Max_model.h5"
            MaxSavePred = './Data/'+date + '_'+ pred_FolderName +'/' + pred_FolderName + '_'+ pred_organ + '_' +"Max_pred.npy"
            Prediction(TestDataName,WeightPath,Save_Max_ModelName_test,MaxSavePred)()     
            AI_pred_path= './Data/'+ date + '_'+ pred_FolderName + '/AI/' + 'AI' + pred_organ 
            predict_to_image(MaxSavePred,pred_FolderName,AI_pred_path)() 
            ui.printf(pred_organ + '_Prediction_Finish')  

        if self.checkBox_6.isChecked():
            pred_organ=self.checkBox_6.text()
            ui.printf(pred_organ + '_Prediction_Start')  
            WeightPath='./weight/Fusion_Kidney_weights.h5'
            Save_Max_ModelName_test = './Data/'+ date + '_'+ pred_FolderName +'/' + pred_FolderName + '_' +pred_organ + '_' +"Max_model.h5"
            MaxSavePred = './Data/'+date + '_'+ pred_FolderName +'/' + pred_FolderName + '_'+ pred_organ + '_' +"Max_pred.npy"
            Prediction(TestDataName,WeightPath,Save_Max_ModelName_test,MaxSavePred)()     
            AI_pred_path= './Data/'+ date + '_'+ pred_FolderName + '/AI/' + 'AI' + pred_organ 
            predict_to_image(MaxSavePred,pred_FolderName,AI_pred_path)() 
            ui.printf(pred_organ + '_Prediction_Finish')    
   
    
    
    
    
        label_path='./Data/'+ date + '_'+ pred_FolderName + '/AI/'
        dicomRT_file = self.lineEdit_2.text()
        save_path = DCM_FolderName + '.dcm'
        self.lineEdit_3.setText(save_path)
        
        dcm_file_path = './'+ DCM_FolderName + '/' 
        itemColorList_path = './Data/'+'Readme.txt' 

 
   #     spacingDatabase, item, DICOMInformation = SegmentiontoImageData(dcm_file_path, label_path)()
        spacingDatabase, item, ROIColor, DICOMInformation,AILabel = SegmentiontoImageData(dcm_file_path, label_path, itemColorList_path)()
        DICOM_RT = pydicom.dcmread(dicomRT_file)
        AI_DICOM_RT = ImagetoRT(spacingDatabase, item, DICOMInformation, DICOM_RT,AILabel)()
        
    
        #   特殊格式
        AI_DICOM_RT.file_meta.MediaStorageSOPInstanceUID = DICOM_RT.SOPInstanceUID + date
        AI_DICOM_RT.SOPInstanceUID = DICOM_RT.SOPInstanceUID + date
        AI_DICOM_RT.file_meta.ImplementationClassUID = '1.3.6.1.4.1.9590.100.1.3.100.9.4'

        pydicom.filewriter.dcmwrite(save_path, AI_DICOM_RT, write_like_original=True)

        AI_DICOM_RT.save_as(save_path)
        ui.printf('ImagetoRT_Finish')    
        self.pushButton_3.setEnabled(False)
        self.pushButton.setEnabled(True)  



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)      #創建一個QApplication，也就是你要開發的軟件app
    window = QtWidgets.QMainWindow()            #創建一個QMainWindow，以便裝載你需要的各種組件，控件 
    ui = Ui_MainWindow()                        #ui是Ui_MainWindow（）類的實例化對象
    ui.setupUi(window)                          #執行類中的setupUi方法，方法的參數是第二步中創建的QMainWindow
    window.show()                               #顯示QMainWindow
    sys.exit(app.exec_())                       #使用exit（）或者單擊關閉按鈕退出QApplication
