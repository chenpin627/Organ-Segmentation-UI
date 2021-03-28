import os
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import pickle
import keras
import cv2
from keras import optimizers
from tqdm import tqdm
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler
import MultiGPUCheckpointCallback as my_config 
from Fusion_model import *
from keras_metrics_loss  import *
import random



class train(object):
    def __init__(self,Train_Data,Save_Weight,organ):
		# input CT or RT data
        self.Train_Data=Train_Data
        self.Save_Weight=Save_Weight
        self.organ=organ
    
    def __call__(self):

        # GPU
        #if don't use mutiple gpus, this is unnecessary
        os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"      #指定使用那塊gpu訓練
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True            #設置動態分配GPU
        session = tf.Session(config=config)
        KTF.set_session(session)
        
              
        # data==================================
        Data = np.load(self.Train_Data)      #載入 Train Data
        TrainData_x = Data['X']            
        TrainData_y = Data['Y']
        
        np.random.seed(7)
        permutation = np.random.permutation(len(TrainData_x))
        TrainData_x = TrainData_x[permutation]
        TrainData_y = TrainData_y[permutation]

        TrainData_x = TrainData_x /255.

        print("===============")
        print('trainData_x : ', TrainData_x.shape)
        print('trainData_y : ', TrainData_y.shape)
        print('shuffle OK')

        
        batch_size = 6
        img_size = 256 
        
        ######## training ########## 
        if self.organ == 'Lung' :
            epochs = 50
            
        elif self.organ == 'Liver' :
            epochs = 115
            
        elif self.organ == 'Stomach' :
            epochs = 115
            
        elif self.organ == 'Esophagus' :
            epochs = 50            
            
        elif self.organ == 'Heart' :
            epochs = 50         
            
        elif self.organ == 'Kidney' :
            epochs = 50       
            
        input_layer = Input(shape=(3, img_size, img_size, 1), name='input_layer')
        Unet_input_layer = Input(shape=(img_size, img_size, 1), name='Unet_input_layer')
        DSS_output = Fusion(input_layer,Unet_input_layer,2)
        DSS = Model(inputs=[input_layer,Unet_input_layer], outputs=DSS_output)

        DSS.summary()
        
        
        # multi gpu
        optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        gpu_model = multi_gpu_model(DSS, gpus=3)
        gpu_model.compile(optimizer=optimizer, loss=[gen_dice_loss_corss_entropy], metrics=[dice_coef]) 
        
         
        history = gpu_model.fit([TrainData_x[:],TrainData_x[:,1]], [ TrainData_y[:,1] ],
                            batch_size = batch_size,
                            validation_data=None,
                            #validation_data = ([TestData_x[:],TestData_x[:,1]], [TestData_y[:,1]]),
                            epochs = epochs,
                            verbose = 1,
                            shuffle = 1,
                            )

        ## save weights and history ######
        print('Save Weights ... ')
        DSS.save_weights(self.Save_Weight)        



class Prediction(object):
    def __init__(self,Test_Data,Weight_path,Save_Model,Save_Pred):
        self.Test_Data=Test_Data
        self.Weight_path = Weight_path
        self.Save_Model = Save_Model
        self.Save_Pred = Save_Pred
    
    def __call__(self):
        # GPU
        #if don't use mutiple gpus, this is unnecessary
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"      #指定使用那塊gpu訓練
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True            #設置動態分配GPU
        session = tf.Session(config=config)
        KTF.set_session(session)


        Test_Data = self.Test_Data
        # data==================================
        Data = np.load(Test_Data)      #載入 Train Data
        TestData_x = Data['X'] 
        
        TestData_x = TestData_x /255.

        print("===============")
        print('TestData_x : ', TestData_x.shape)
        print('shuffle OK')

        img_size = 256 

        input_layer = Input(shape=(3, img_size, img_size, 1), name='input_layer')
        Unet_input_layer = Input(shape=(img_size, img_size, 1), name='Unet_input_layer')
        DSS_output = Fusion(input_layer,Unet_input_layer,2)
        DSS = Model(inputs=[input_layer,Unet_input_layer], outputs=DSS_output)
        DSS.summary()


        # multi gpu
# =============================================================================
#         optimizer = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#         gpu_model = multi_gpu_model(DSS, gpus=3)
#         gpu_model.compile(optimizer=optimizer, loss=[gen_dice_loss_corss_entropy], metrics=[dice_coef])          
# =============================================================================



        DSS.load_weights(self.Weight_path)   

        
        ######## testing ##########
        for i in range(1):
            print("#################" + str(i) + "#################")
            y_pred = DSS.predict([TestData_x[:], TestData_x[:,1]], batch_size = 6, verbose=1)
            DSS.save(self.Save_Model)
            np.save(self.Save_Pred  , y_pred)       

























