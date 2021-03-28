import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os

def Dice_1(y_true, y_pred):
    dice = K.cast(0, dtype='float32')
    true = K.argmax(y_true,axis=-1)
    true = K.cast(K.equal(true, 1), dtype='float32')

    pred = K.argmax(y_pred, axis=-1)
    pred = K.cast(K.equal(pred, 1), dtype='float32')

    num =  K.sum(true * pred)
    den =  K.sum(pred) + K.sum(true)
    den = K.switch(K.equal(den,0), K.epsilon(), den)
    dice = K.clip(  (2. * num)/den, K.epsilon(), 1-K.epsilon())

    # dice = K.mean(num / den, axis=0)
    # x = K.cast(K.int_shape(y_pred)[0],dtype='float32')

    return dice


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

class Metrics(keras.callbacks.Callback):

    def __init__(self, filepath, base_model, validation_data_x, validation_data_y, thresh, metricsName, npyName):
        super(Metrics, self).__init__()
        self.val_dice = 0
        self.base_model = base_model
        # self.x = validation_data_x
        # self.y = validation_data_y
        self.filepath = filepath
        self.thresh = thresh
        self.epochs_total_times = 0  
        self.save_best_only = 1
        self.metricsName = 'Val_Dice'
        self.npyName = npyName
         
        self.monitor_op = np.greater
        self.best = -np.Inf



    def on_epoch_end(self, epoch, logs={}):
        logs = logs or {}
        self.epochs_total_times += 1

        val_predict = 0
        val_targ = 0
        self.total_DICE = 0


        # dice
        if self.epochs_total_times >= self.thresh:
            val_predict = (np.asarray(self.base_model.predict(self.validation_data[0], batch_size=6)))

            val_targ = self.validation_data[1]
            val_predict = np.equal( np.argmax(val_predict, axis=-1), 1)
            val_targ = np.equal( np.argmax(val_targ, axis=-1), 1)
            ones = np.ones(np.shape(val_targ))
            TP = np.sum(val_predict * val_targ)
            FP = np.sum((ones - val_targ) * val_predict )
            FN = np.sum((ones - val_predict) * val_targ)
            DICE =  (2. * TP) / (2.*TP+FP+FN )
            # self.total_DICE = self.total_DICE + DICE
            print('Val_Dice :', DICE)
        

        # check point
        if self.epochs_total_times >= self.thresh:
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = DICE
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                'skipping.' % (self.metricsName), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                            ' saving model to %s'
                                % (epoch + 1, self.metricsName, self.best,
                                    current, filepath) )
                        self.best = current
                        self.base_model.save_weights(filepath, overwrite=True)
                        np.save(self.npyName, val_predict)
                    else: 
                        print('Epoch %05d: %s did not improve' %
                                    (epoch + 1, self.metricsName))
        del val_predict, val_targ
        return
        

class MultiGPUCheckpointCallback(keras.callbacks.Callback):

    def __init__(self, filepath, base_model, monitor='val_IoU_1', verbose=1,
                 save_best_only=True, save_weights_only=True,
                 mode='auto', period=25, thresh = 125):
        super(MultiGPUCheckpointCallback, self).__init__()
        self.base_model = base_model
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.epochs_total_times = 0 
        self.thresh = thresh

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        self.epochs_total_times += 1 
        if self.epochs_total_times >= self.thresh:
            if self.epochs_since_last_save >= self.period:
                self.epochs_since_last_save = 0
                filepath = self.filepath.format(epoch=epoch + 1, **logs)
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        warnings.warn('Can save best model only with %s available, '
                                      'skipping.' % (self.monitor), RuntimeWarning)
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            if self.save_weights_only:
                                self.base_model.save_weights(filepath, overwrite=True)
                            else:
                                self.base_model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('Epoch %05d: %s did not improve' %
                                      (epoch + 1, self.monitor))
                else:
                    if self.verbose > 0:
                        print('Epoch %05d: saving model to %s' % (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.base_model.save_weights(filepath, overwrite=True)
                    else:
                        self.base_model.save(filepath, overwrite=True)

def test_npy(npy_path):
    x = np.load(npy_path)
    x = np.argmax(x,axis=-1)

    lung_pixel = np.where(x==1)
    liverg_pixel = np.where(x==2)
    eso_pixel = np.where(x==3)
    x = np.expand_dims(x,axis=-1)
    x = np.array( np.repeat(x, 3, axis=-1), dtype=np.uint8)
    print(x.shape)
    x[lung_pixel] = [193,193,255]
    x[liverg_pixel] =[139,236,255]
    x[eso_pixel] = [127, 255, 0]

    for i in range(len(x)):
        cv2.imwrite('./4class/' + '{:04}'.format(i) + '.bmp', x[i])

# import cv2
# test_npy('./Unet_4class_60p_0713_Max.npy')


