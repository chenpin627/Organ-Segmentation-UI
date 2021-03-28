
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



def BN_relu(tensor, layerName):
	tensor = BatchNormalization(name=layerName + "BN")(tensor)
	tensor = Activation('relu', name=layerName + "Act")(tensor)
	return tensor



def AttentionBlock_timedisbuted(de, en,stage):
    layerName = 'Attent_' + str(stage) +'_'
    # resnet
    shape_en = K.int_shape(en)  
    # deconv
    shape_de = K.int_shape(de)  

    en_conv1x1 =TimeDistributed(Conv2D(1, kernel_size=(1,1), strides=(1,1), use_bias=False, padding='same'), name=layerName+'en_conv1x1')(en)
    en_conv1x1_bn = BatchNormalization(name=layerName+'en_conv1x1_bn')(en_conv1x1)

    de_conv1x1 = TimeDistributed(Conv2D(1, kernel_size=(1,1), strides=(1,1), use_bias=False, padding='same'), name=layerName+'de_conv1x1')(de)
    de_conv1x1_bn = BatchNormalization(name=layerName+'de_conv1x1bn')(de_conv1x1)

    concat = keras.layers.add([en_conv1x1_bn, de_conv1x1_bn] )
    concat_relu = Activation('relu')(concat)


    fusion = TimeDistributed(Conv2D(1, kernel_size=(1,1), strides=(1,1), use_bias=False, padding='same'), name=layerName+'fusion_conv1x1')(concat_relu)
    fusion_conv1x1_bn = BatchNormalization(name=layerName+'fusion_conv1x1bn')(fusion)
    fusion_conv1x1_bn = Activation('sigmoid')(fusion_conv1x1_bn)
    # UpSampling = UpSampling2D(size=(2, 2), interpolation='bilinear',name=layerName + 'up')(fusion_conv1x1_bn)

    output = multiply([fusion_conv1x1_bn, en])
    return output


def Unet(input_layer, output_class):

    Conv1 = Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='conv_1-')(input_layer)
    act1 = BN_relu(Conv1, layerName='conv_1_-')
    Conv2 = Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='conv_2-')(act1)
    act2 = BN_relu(Conv2, layerName='conv_2_-')

    MaxPooling1 = MaxPooling2D(pool_size=(2,2), name='max_pool_1-')(act2)

    Conv3 = Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='conv_3-')(MaxPooling1)
    act3 = BN_relu(Conv3, layerName='conv_3_-')
    Conv4 = Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='conv_4-')(act3)
    act4 = BN_relu(Conv4, layerName='conv_4_-')

    MaxPooling2 = MaxPooling2D(pool_size=(2,2), name='max_pool_2-')(act4)

    Conv5 = Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='conv_5-')(MaxPooling2)
    act5 = BN_relu(Conv5, layerName='conv_5_-')
    Conv6 = Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='conv_6-')(act5)
    act6 = BN_relu(Conv6, layerName='conv_6_-')
    MaxPooling3 = MaxPooling2D(pool_size=(2,2), name='max_pool_3-')(act6)

    Conv7 = Conv2D(512, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='conv_7-')(MaxPooling3)
    act7 = BN_relu(Conv7, layerName='Conv7_-')
    Conv8 = Conv2D(512, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv8-')(act7)
    act8 = BN_relu(Conv8, layerName='Conv8_-')
    Conv8_ = Conv2D(512, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv8_2-')(act8)
    act8_ = BN_relu(Conv8_, layerName='Conv8_2_-')

    UpSampling1 = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up1-')(act8_)
    Concat1 = concatenate(([act6, UpSampling1]), axis=-1)

    Conv9 = Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv9-')(Concat1)
    act9 = BN_relu(Conv9, layerName='Conv9_-')
    Conv10 = Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv10-')(act9)
    act10 = BN_relu(Conv10, layerName='Conv10_-')


    UpSampling2 = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up2-')(act10)
    Concat2 = concatenate(([act4, UpSampling2]), axis=-1)

    Conv11 = Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv11-')(Concat2)
    act11 = BN_relu(Conv11, layerName='Conv11_-')
    Conv12= Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv12-')(act11)
    act12 = BN_relu(Conv12, layerName='Conv12_-')


    UpSampling3 = UpSampling2D(size=(2, 2), interpolation='bilinear', name='up3-')(act12)
    Concat3 = concatenate(([act2, UpSampling3]), axis=-1)

    Conv13 = Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv13-')(Concat3)
    act13 = BN_relu(Conv13, layerName='Conv13_-')
    Conv14 = Conv2D(32, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='Conv14-')(act13)
    act14 = BN_relu(Conv14, layerName='Conv14_-')


    Output = Conv2D(output_class, kernel_size=(1,1), strides=(1,1), padding='same', name='output-')(act14)
    Output = Activation('softmax', name='output_act-')(Output)
    
    return Output

def Fusion(input_layer, Unet_input_layer, class_num):

    Conv1 = TimeDistributed(Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='conv_1')(input_layer)
    act1 = BN_relu(Conv1, layerName='conv_1_')
    Conv2 = TimeDistributed(Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='conv_2')(act1)
    act2 = BN_relu(Conv2, layerName='conv_2_')

    MaxPooling1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name='max_pool_1')(act2)

    Conv3 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='conv_3')(MaxPooling1)
    act3 = BN_relu(Conv3, layerName='conv_3_')
    Conv4 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='conv_4')(act3)
    act4 = BN_relu(Conv4, layerName='conv_4_')

    MaxPooling2 = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name='max_pool_2')(act4)

    Conv5 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='conv_5')(MaxPooling2)
    act5 = BN_relu(Conv5, layerName='conv_5_')
    Conv6 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='conv_6')(act5)
    act6 = BN_relu(Conv6, layerName='conv_6_')
    MaxPooling3 = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name='max_pool_3')(act6)

    BCLSTM1 = Bidirectional(ConvLSTM2D(486, kernel_size=(3,3), strides=(1, 1), padding="same",return_sequences=True, stateful=False), merge_mode='sum', name='BCLSTM1' )(MaxPooling3)
    BCLSTM1_bn = BatchNormalization(name='BCLSTM1_bn' )(BCLSTM1)
    BCLSTM1_relu = Activation('relu')(BCLSTM1_bn)

    UpSampling1 = TimeDistributed(UpSampling2D(size=(2, 2), interpolation='bilinear'), name='up1')(BCLSTM1_relu)
    AttentionBlock_1 = AttentionBlock_timedisbuted(UpSampling1, act6, '1')
    Concat1 = concatenate(([AttentionBlock_1, UpSampling1]), axis=-1)

    Conv7 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='conv_7')(Concat1)
    act7 = BN_relu(Conv7, layerName='Conv7_')
    Conv8 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='Conv8')(act7)
    act8 = BN_relu(Conv8, layerName='Conv8_')

    UpSampling2 = TimeDistributed(UpSampling2D(size=(2, 2), interpolation='bilinear'), name='up2')(act8)
    AttentionBlock_2 = AttentionBlock_timedisbuted(UpSampling2, act4, '2')
    Concat2 = concatenate(([AttentionBlock_2, UpSampling2]), axis=-1)

    Conv9 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='Conv9')(Concat2)
    act9 = BN_relu(Conv9, layerName='Conv9_')
    Conv10 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='Conv10')(act9)
    act10 = BN_relu(Conv10, layerName='Conv10_')

    UpSampling3 = TimeDistributed(UpSampling2D(size=(2, 2), interpolation='bilinear'), name='up3')(act10)
    AttentionBlock_3 = AttentionBlock_timedisbuted(UpSampling3, act2, '3')
    Concat3 = concatenate(([AttentionBlock_3, UpSampling3]), axis=-1)

    Conv11 = TimeDistributed(Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='Conv11')(Concat3)
    act11 = BN_relu(Conv11, layerName='Conv11_')

    B_CLSTM_last = Bidirectional(ConvLSTM2D(64, kernel_size=(3,3), strides=(1,1), padding="same",return_sequences=False, stateful=False), merge_mode='sum', name='B_CLSTM_last' )(act11)
    B_CLSTM_last_ = BN_relu(B_CLSTM_last, layerName='B_CLSTM_last_')  

    Output1 = Conv2D(class_num, kernel_size=(1,1), strides=(1,1), padding='same', name='output')(B_CLSTM_last_)
    # Output1 = TimeDistributed(Conv2D(class_num, kernel_size=(1,1), strides=(1,1), use_bias=False, padding='same'), name='output')(B_CLSTM_last_)
    Output1 = Activation('softmax', name='output_act')(Output1)

    stage_concate = multiply([Output1, Unet_input_layer])
    # final_fusion = concatenate(([stage_concate, Unet_input_layer]), axis=-1)
    final_fusion = keras.layers.add([stage_concate, Unet_input_layer] )
    Unet_output = Unet(final_fusion, 2)
    # final_fusion = concatenate(([Output1, Unet_output]), axis=-1)
    # final_Conv = Conv2D(32, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same', name='final_Conv')(final_fusion)
    # final_act = BN_relu(final_Conv, layerName='final_Conv_')
    # final_Output = Conv2D(class_num, kernel_size=(1,1), strides=(1,1), use_bias=False, padding='same',name='output_final')(final_act)
    # final_Output = Activation('softmax', name='output_final_fusion')(final_Output)

    # input_layer
    # stage_concate = multiply([Output1, input_layer])




    # # stage2
    # stage2_Conv1 = TimeDistributed(Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_conv_1')(stage_concate)
    # stage2_act1 = BN_relu(stage2_Conv1, layerName='stage2_conv_1_')
    # stage2_Conv2 = TimeDistributed(Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_conv_2')(stage2_act1)
    # stage2_act2 = BN_relu(stage2_Conv2, layerName='stage2_conv_2_')

    # stage2_MaxPooling1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name='stage2_max_pool_1')(stage2_act2)

    # stage2_Conv3 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_conv_3')(stage2_MaxPooling1)
    # stage2_act3 = BN_relu(stage2_Conv3, layerName='stage2_conv_3_')
    # stage2_Conv4 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_conv_4')(stage2_act3)
    # stage2_act4 = BN_relu(stage2_Conv4, layerName='stage2_conv_4_')

    # stage2_MaxPooling2 = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name='stage2_max_pool_2')(stage2_act4)

    # stage2_Conv5 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_conv_5')(stage2_MaxPooling2)
    # stage2_act5 = BN_relu(stage2_Conv5, layerName='stage2_conv_5_')
    # stage2_Conv6 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_conv_6')(stage2_act5)
    # stage2_act6 = BN_relu(stage2_Conv6, layerName='stage2_conv_6_')
    # stage2_MaxPooling3 = TimeDistributed(MaxPooling2D(pool_size=(2,2)), name='stage2_max_pool_3')(stage2_act6)


    # stage2_Conv_mid1 = TimeDistributed(Conv2D(512, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_Conv_mid1')(stage2_MaxPooling3)
    # stage2_Conv_mid1_ = BN_relu(stage2_Conv_mid1, layerName='stage2_Conv_mid1_')
    # stage2_Conv_mid2 = TimeDistributed(Conv2D(512, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_Conv_mid2')(stage2_Conv_mid1_)
    # stage2_Conv_mid2_ = BN_relu(stage2_Conv_mid2, layerName='stage2_Conv_mid2_')
    # stage2_Conv_mid3 = TimeDistributed(Conv2D(512, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_Conv_mid3')(stage2_Conv_mid2_)
    # stage2_Conv_mid3_ = BN_relu(stage2_Conv_mid3, layerName='stage2_Conv_mid3_')

    # stage2_UpSampling1 = TimeDistributed(UpSampling2D(size=(2, 2), interpolation='bilinear'), name='stage2_up1')(stage2_Conv_mid3_)

    # stage2_concate1 = concatenate(([stage2_act6, stage2_UpSampling1]), axis=-1)  

    # stage2_Conv7 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_conv_7')(stage2_concate1)
    # stage2_act7 = BN_relu(stage2_Conv7, layerName='stage2_Conv7_')
    # stage2_Conv8 = TimeDistributed(Conv2D(256, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_Conv8')(stage2_act7)
    # stage2_act8 = BN_relu(stage2_Conv8, layerName='stage2_Conv8_')

    # stage2_UpSampling2 = TimeDistributed(UpSampling2D(size=(2, 2), interpolation='bilinear'), name='stage2_up2')(stage2_act8)

    # stage2_concate2 = concatenate(([stage2_act4, stage2_UpSampling2]), axis=-1)


    # stage2_Conv9 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_Conv9')(stage2_concate2)
    # stage2_act9 = BN_relu(stage2_Conv9, layerName='stage2_Conv9_')
    # stage2_Conv10 = TimeDistributed(Conv2D(128, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_Conv10')(stage2_act9)
    # stage2_act10 = BN_relu(stage2_Conv10, layerName='stage2_Conv10_')

    # stage2_UpSampling3 = TimeDistributed(UpSampling2D(size=(2, 2), interpolation='bilinear'), name='stage2_up3')(stage2_act10)

    # stage2_concate3 = concatenate(([stage2_act2, stage2_UpSampling3]), axis=-1)


    # stage2_Conv11 = TimeDistributed(Conv2D(64, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='stage2_Conv11')(stage2_concate3)
    # stage2_act11 = BN_relu(stage2_Conv11, layerName='stage2_Conv11_')

    # stage2_B_CLSTM_last = Bidirectional(ConvLSTM2D(32, kernel_size=(3,3), strides=(1,1), padding="same",return_sequences=False, stateful=False), merge_mode='sum', name='stage2_B_CLSTM_last' )(stage2_act11)
    # stage2_B_CLSTM_last_ = BN_relu(stage2_B_CLSTM_last, layerName='stage2_B_CLSTM_last_')  

    # stage2_Output = Conv2D(class_num, kernel_size=(1,1), strides=(1,1), padding='same', name='stage2_output')(stage2_B_CLSTM_last_)
    # # stage2_Output = TimeDistributed(Conv2D(class_num, kernel_size=(1,1), strides=(1,1), use_bias=False, padding='same'), name='stage2_output')(stage2_B_CLSTM_last_)
    # # stage2_Output = BatchNormalization()(stage2_Output)
    # Output2 = Activation('softmax', name='stage2_output_act')(stage2_Output)


    # final_fusion = concatenate(([Output1, Output2]), axis=-1)
    # final_Conv = TimeDistributed(Conv2D(32, kernel_size=(3,3), strides=(1,1), use_bias=False, padding='same'), name='final_Conv')(final_fusion)
    # final_act = BN_relu(final_Conv, layerName='final_Conv_')
    # B_CLSTM_fusion = Bidirectional(ConvLSTM2D(32, kernel_size=(3,3), strides=(1,1), padding="same",return_sequences=False, stateful=False), merge_mode='sum', name='B_CLSTM_fusion' )(final_act)
    # B_CLSTM_fusion_ = BN_relu(B_CLSTM_fusion, layerName='B_CLSTM_fusion_')
    # final_Output = Conv2D(class_num, kernel_size=(1,1), strides=(1,1), use_bias=False, padding='same',name='output_final')(B_CLSTM_fusion_)
    # final_Output = Activation('softmax', name='output_final_fusion')(final_Output)

    return Unet_output

# input_layer = Input(batch_shape=(1,3, 256, 256, 1))
# Unet_input_layer = Input(batch_shape=(1,256, 256, 1))
# DSS_output1 = Fusion(input_layer,Unet_input_layer, 2)
# DSS = Model(inputs=[input_layer,Unet_input_layer], outputs=DSS_output1)
# DSS.summary()


# from keras.utils import plot_model
# from keras.callbacks import TensorBoard
# plot_model(DSS, to_file='test2.png', show_shapes=True)


# one output
# input_layer = Input(shape=(256, 256, 1))
# DSS_output = AUnet(input_layer,2)
# DSS = Model(inputs=input_layer, outputs=DSS_output)
# DSS.summary()


def save_img(npy):
    import cv2
    x = np.load(npy)
    x = np.array(np.argmax(x,axis=-1), dtype=np.uint8)*255
    for i in range(len(x)):
        cv2.imwrite('./img/' + '{:04}'.format(i) + '.bmp', x[i])
# save_img('0729_Fusion_4_stomach__Max_pred.npy')