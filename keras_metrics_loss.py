
import numpy as np
import keras 
import tensorflow as tf
from keras import backend as K
from keras import Sequential
from keras.engine import Input, Model
from keras.layers import Conv3D,Conv2D, MaxPooling3D, UpSampling2D, UpSampling3D, Activation, BatchNormalization, PReLU, Lambda, GlobalAveragePooling2D, GlobalAveragePooling3D,merge
from keras.layers import TimeDistributed, MaxPooling2D, ConvLSTM2D, Bidirectional, Concatenate, Conv2DTranspose, Dense, AveragePooling2D, Dropout,concatenate,  Reshape, Dense, multiply, Permute
from keras.optimizers import Adam, RMSprop
from keras.layers.normalization import BatchNormalization

def dice_coef(y_true, y_pred):
    true = K.argmax(y_true,axis=-1)
    true = K.equal(true, 1)
    true = K.cast(K.reshape(true, [-1]),dtype ='float32')

    pred = K.argmax(y_pred,axis=-1)
    pred = K.equal(pred, 1)
    pred = K.cast(K.reshape(pred, [-1]),dtype ='float32')

    intersection = K.sum(true * pred)
    return (2. * intersection + K.epsilon()) / (K.sum(true) + K.sum(pred) + K.epsilon())


def gen_dice_loss_corss_entropy(y_true, y_pred):
    '''
    computes the sum of two losses : generalised dice loss 
    '''

    #generalised dice score is calculated as in this paper : https://arxiv.org/pdf/1707.03237


    y_true_f = K.reshape(y_true,shape=(-1,2))
    y_pred_f = K.reshape(y_pred,shape=(-1,2))
    sum_p=K.sum(y_pred_f,axis=-2)
    sum_r=K.sum(y_true_f,axis=-2)
    sum_pr=K.sum(y_true_f * y_pred_f,axis=-2)
    weights=K.pow(K.square(sum_r)+K.epsilon(),-1)
    generalised_dice_numerator =2*K.sum(weights*sum_pr)
    generalised_dice_denominator =K.sum(weights*(sum_r+sum_p))
    generalised_dice_score =generalised_dice_numerator /generalised_dice_denominator
    GDL=1-generalised_dice_score
    del sum_p,sum_r,sum_pr,weights
    # return GDL
    return GDL


def ConfusionMatrix(y_true, y_pred, class_num):
    classes = y_true.shape[-1] - 1
    meanIoU = 0
    meanDice = 0
    meanKappa = 0
    Liver_Dice = 0
    Stomach_Dice = 0
    ones = np.ones(np.shape(y_true[...,1]))

    for i in range(1,class_num): 
        pred = np.equal( np.argmax(y_pred, axis=-1), i)
        inv_pred = ones - pred
        true = np.equal( np.argmax(y_true, axis=-1), i) 
        inv_true = ones - true

        TP = np.sum(pred * true)
        FP = np.sum(pred * inv_true)
        FN = np.sum(inv_pred * true)
        TN = np.sum(inv_pred * inv_true)
        N =  TP + FP + FN + TN   
        Dice = (2*TP) / (2*TP+FP+FN)

        # if i==1:
        #     Stomach_Dice =  Dice
        print("=== class " + str(i) + " ===")   
        print("TP: ", TP)
        print("FP: ", FP)
        print("FN: ", FN)
        print("TN: ", TN)
        print("Dice: ", Dice ) 
        meanDice = meanDice + Dice

    print("=== mean ===")
    print("Mean_Dice: ", meanDice / classes)
    print("############")
    return meanDice