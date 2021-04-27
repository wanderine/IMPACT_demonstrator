# conda activate keras2

import argparse

import os
import warnings
import numpy as np
#from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from utilities import load_data
from utilities import load_data_autoscale

import nibabel as nib 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0";

warnings.simplefilter(action='ignore',  category=FutureWarning)

from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K

class unet(object):
    
    def __init__(self, img_size, Nclasses, class_weights, model_name='myWeightsAug.h5', Nfilter_start=64, depth=3, batch_size=3):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        self.model_name = model_name
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        self.batch_size = batch_size

        self.model = Sequential()
        inputs = Input(img_size)
    
        def dice(y_true, y_pred, eps=1e-5):
            num = 2.*K.sum(self.class_weights*K.sum(y_true * y_pred, axis=[0,1,2]))
            den = K.sum(self.class_weights*K.sum(y_true + y_pred, axis=[0,1,2]))+eps
            return num/den

        def diceLoss(y_true, y_pred):
            return 1-dice(y_true, y_pred)       
    
        def bceLoss(y_true, y_pred):
            bce = K.sum(-self.class_weights*K.sum(y_true*K.log(y_pred), axis=[0,1,2]))
            return bce    
        
        # This is a help function that performs 2 convolutions, each followed by batch normalization
        # and ReLu activations, Nf is the number of filters, filter size (3 x 3)
        def convs(layer, Nf):
            x = Conv2D(Nf, (3,3), kernel_initializer='he_normal', padding='same')(layer)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(Nf, (3,3), kernel_initializer='he_normal', padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
            
        # This is a help function that defines what happens in each layer of the encoder (downstream),
        # which calls "convs" and then Maxpooling (2 x 2). Save each layer for later concatenation in the upstream.
        def encoder_step(layer, Nf):
            y = convs(layer, Nf)
            x = MaxPooling2D(pool_size=(2,2))(y)
            return y, x
            
        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains transpose convolution (filter size (3 x 3), stride (2,2) batch normalization, concatenation with 
        # corresponding layer (y) from encoder, and lastly "convs"
        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv2DTranspose(filters=Nf, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_normal')(layer)
            x = BatchNormalization()(x)
            x = concatenate([x, layer_to_concatenate])
            x = convs(x, Nf)
            return x
            
        layers_to_concatenate = []
        x = inputs
        
        # Make encoder with 'self.depth' layers, 
        # note that the number of filters in each layer will double compared to the previous "step" in the encoder
        for d in range(self.depth-1):
            y,x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(y)
            
        # Make bridge, that connects encoder and decoder using "convs" between them. 
        # Use Dropout before and after the bridge, for regularization. Use dropout probability of 0.2.
        x = Dropout(0.2)(x)
        x = convs(x,self.Nfilter_start*np.power(2,self.depth-1))
        x = Dropout(0.2)(x)        
        
        # Make decoder with 'self.depth' layers, 
        # note that the number of filters in each layer will be halved compared to the previous "step" in the decoder
        for d in range(self.depth-2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start*np.power(2,d))            
            
        # Make classification (segmentation) of each pixel, using convolution with 1 x 1 filter
        final = Conv2D(filters=self.Nclasses, kernel_size=(1,1), activation = 'softmax')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(loss=diceLoss, optimizer=Adam(lr=1e-4), metrics=['accuracy',dice]) 
        
    def train(self, X, Y, x, y, nEpochs):
        print('Training process:')
        callbacks = [ModelCheckpoint(self.model_name, verbose=0, save_best_only=True, save_weights_only=True),
                    EarlyStopping(patience=20)]
        results = self.model.fit(X, Y, validation_data=(x,y), batch_size=self.batch_size, epochs=nEpochs, callbacks=callbacks, workers=4, use_multiprocessing=True)
        return results
        
    def train_with_aug(self, im_gen_train, gt_gen_train, im_gen_valid, gt_gen_valid, nEpochs):       
        print('Training process:')
        # we save in a dictionary the metricts obtained after each epoch
        results_dict = {}
        results_dict['loss'] = []
        results_dict['accuracy'] = []
        results_dict['dice'] = []
        results_dict['val_loss'] = []
        results_dict['val_accuracy'] = []
        results_dict['val_dice'] = []
        
        val_loss0 = np.inf
        steps_val_not_improved = 0
        for e in range(nEpochs):
            print('\nEpoch {}/{}'.format(e+1, nEpochs))
            
            Xb_train, Yb_train = im_gen_train.next(), gt_gen_train.next()
            Xb_valid, Yb_valid = im_gen_valid.next(), gt_gen_valid.next()

            # Transform ground truth images into categorical
            Yb_train = to_categorical(np.argmax(Yb_train, axis=-1), self.Nclasses)
            Yb_valid = to_categorical(np.argmax(Yb_valid, axis=-1), self.Nclasses)               

            results = self.model.fit(Xb_train, Yb_train, validation_data=(Xb_valid,Yb_valid), batch_size=self.batch_size)
            #results = self.model.fit(Xb_train, Yb_train, validation_data=(Xb_valid,Yb_valid), batch_size=self.batch_size, workers=4, use_multiprocessing=True)

            if results.history['val_loss'][0] <= val_loss0:
                self.model.save_weights(self.model_name)
                print('val_loss decreased from {:.4f} to {:.4f}. Hence, new weights are now saved in {}.'.format(val_loss0, results.history['val_loss'][0], self.model_name))
                val_loss0 = results.history['val_loss'][0]
                steps_val_not_improved = 0
            else:
                print('val_loss did not improve.')
                steps_val_not_improved += 1

            # saving the metrics
            results_dict['loss'].append(results.history['loss'][0])
            results_dict['accuracy'].append(results.history['accuracy'][0])
            results_dict['dice'].append(results.history['dice'][0])
            results_dict['val_loss'].append(results.history['val_loss'][0])
            results_dict['val_accuracy'].append(results.history['val_accuracy'][0])
            results_dict['val_dice'].append(results.history['val_dice'][0])
            
            if steps_val_not_improved==20:
                print('\nThe training stopped because the network after 20 epochs did not decrease it''s validation loss.')
                break

        return results_dict
    
    def evaluate(self, X, Y):
        print('Evaluation process:')
        score, acc, dice = self.model.evaluate(X,Y,self.batch_size)
        print('Accuracy: {:.4f}'.format(acc*100))
        print('Dice: {:.4f}'.format(dice*100))
        return acc, dice
    
    def predict(self, X):
        print('Segmenting unseen image')
        segmentation = self.model.predict(X,self.batch_size)
        return segmentation

  

def segment(input_path: str, output_segmentation: str):

    mymax_T1GD = 9331.0 / 2.0

    mymax_qMRIT1 = 6252.2 / 2.0
    mymax_qMRIT2 = 3172.6 / 2.0 
    mymax_qMRIPD = 183.0 / 2.0

    mymax_qMRIT1_GD = 6405.2 / 2.0
    mymax_qMRIT2_GD = 3070.6 / 2.0
    mymax_qMRIPD_GD = 189.07 / 2.0

    anatomical = nib.load(input_path + "/T1GD/volume.nii") # 9_t1w_gd.nii.gz")
    new_header = anatomical.header.copy() 
    
    # Check number of folders
    folders = 0
    for _, dirnames, filenames in os.walk(input_path):
        # ^ this idiom means "we won't be using this value"        
        folders += len(dirnames)

    if folders == 1:    
        testImages = load_data_autoscale(data_directory=input_path + "/T1GD/", nr_to_load=1, maxintensity=mymax_T1GD) 
        modelName = '/home/myWeights_weight60000_depth4_nfilter16_CV3_BRATS_augmented_defaced.h5'
        print("Only using T1GD")
    elif folders == 4:
        images1 = load_data_autoscale(data_directory=input_path + "/T1GD/", nr_to_load=1, maxintensity=mymax_T1GD)
        images2 = load_data(data_directory=input_path + "/qMRIT1GD/", nr_to_load=1, maxintensity=mymax_qMRIT1_GD)
        images3 = load_data(data_directory=input_path + "/qMRIT2GD/", nr_to_load=1, maxintensity=mymax_qMRIT2_GD)
        images4 = load_data(data_directory=input_path + "/qMRIPDGD/", nr_to_load=1, maxintensity=mymax_qMRIPD_GD)
        testImages = np.concatenate((images1,images2,images3,images4),axis=3)
        modelName = '/home/myWeights_weight60000_depth4_nfilter16_CV3_BRATS_qMRIGD_augmented_defaced.h5'
        print("Using T1GD and qMRI GD")
    elif folders == 7:
        images1 = load_data_autoscale(data_directory=input_path + "/T1GD/", nr_to_load=1, maxintensity=mymax_T1GD)
        images2 = load_data(data_directory=input_path + "/qMRIT1/", nr_to_load=1, maxintensity=mymax_qMRIT1)
        images3 = load_data(data_directory=input_path + "/qMRIT2/", nr_to_load=1, maxintensity=mymax_qMRIT2)
        images4 = load_data(data_directory=input_path + "/qMRIPD/", nr_to_load=1, maxintensity=mymax_qMRIPD)
        images5 = load_data(data_directory=input_path + "/qMRIT1GD/", nr_to_load=1, maxintensity=mymax_qMRIT1_GD)
        images6 = load_data(data_directory=input_path + "/qMRIT2GD/", nr_to_load=1, maxintensity=mymax_qMRIT2_GD)
        images7 = load_data(data_directory=input_path + "/qMRIPDGD/", nr_to_load=1, maxintensity=mymax_qMRIPD_GD)
        testImages = np.concatenate((images1,images2,images3,images4,images5,images6,images7),axis=3)
        modelName = '/home/myWeights_weight60000_depth4_nfilter16_CV3_BRATS_qMRI_qMRIGD_augmented_defaced.h5'
        print("Using T1GD and qMRI and qMRI GD")

    print(testImages.shape)     

    img_size = testImages[0].shape
    Nclasses = 2
    class_weights = np.zeros((2,1))
    class_weights[0] = 1
    class_weights[1] = 60000
    net_aug = unet(img_size, Nclasses, class_weights, modelName, Nfilter_start=16, batch_size=4, depth=4)

    net_aug.model.load_weights(modelName)
    segmentation = net_aug.predict(testImages)
    segmentation = segmentation.transpose(1,2,0,3)
    segmentation = np.argmax(segmentation, axis=-1)
    segmentation = segmentation.astype(np.float32)
    segmentation[segmentation < 0.5] = 0
    segmentation[segmentation >= 0.5] = 1
    img = nib.Nifti1Image(segmentation, None, header=new_header)
    img.set_data_dtype(np.float32)
    nib.save(img, output_segmentation) 
    

def get_parser():
    """
    Parse input arguments.
    """
    parser = argparse.ArgumentParser(description='Tumor segmentation')

    # Positional arguments.
    parser.add_argument("input_image_path", help="Path to input image(s)")
    parser.add_argument("output_segmentation", help="Name of segmentation output image (Nifti)")
    return parser.parse_args()

if __name__ == "__main__":
    p = get_parser()

    segment(p.input_image_path, p.output_segmentation)



