import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps

import os

from sklearn.utils import class_weight

import keras 
from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, AveragePooling2D, Flatten, GlobalMaxPool2D, Input, BatchNormalization
from keras.layers import Activation
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator


from models import Model_1, build_model

import deepdish as dd

global base_dir
base_dir = 'prepared_data/'

def prepare_data(config):
    img_width, img_height = config['img_width'], config['img_height']
    base_dir = config['base_dir']
    print(base_dir)
    
        
    df = pd.read_csv('HAM10000_metadata.csv')
    if config['TEST'] is not None:
        print(config['TEST'])
        base_dir = 'prepared_data_test/'
        df = df[0:10]

    df['age'] = df['age'].fillna(df['age'].mean())
    df['img_path'] = df['image_id'].apply(lambda x : 'data/{}.jpg'.format(x))

    img_data = np.empty((len(df), img_height, img_width, 3))
    #Resize images and save
    for i,v in enumerate(df['img_path']):
        print('{} / {}'.format(i, len(df)))
        v2 = np.asarray(ImageOps.fit(Image.open(v), (img_width,img_height))).astype('float32')
        img_data[i] = v2
    
    np.save('resized_img_data.npy', img_data)
    feature_data = np.asarray(pd.get_dummies(df[['dx_type','age','sex','localization']]))
    target_data = np.asarray(pd.get_dummies(df['dx']))
    

    del df
    

    np.random.seed(42)
    r_ind = np.random.rand(len(img_data))
    train_ind = r_ind < 0.7
    test_ind = (r_ind>=0.7) & (r_ind < 0.85)
    val_ind = r_ind >= 0.85

    x_im_train = img_data[train_ind]
    x_im_val = img_data[val_ind]
    x_im_test = img_data[test_ind]

    x_feat_train = feature_data[train_ind]
    x_feat_val = feature_data[val_ind]
    x_feat_test = feature_data[test_ind]

    y_train = target_data[train_ind]
    y_val = target_data[val_ind]
    y_test = target_data[test_ind]

    
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    

    np.save('{}x_im_train.npy'.format(base_dir), x_im_train)
    np.save('{}x_im_val.npy'.format(base_dir), x_im_val)
    np.save('{}x_im_test.npy'.format(base_dir), x_im_test)

    np.save('{}x_feat_train.npy'.format(base_dir), x_feat_train)
    np.save('{}x_feat_val.npy'.format(base_dir), x_feat_val)
    np.save('{}x_feat_test.npy'.format(base_dir), x_feat_test)

    np.save('{}y_train.npy'.format(base_dir), y_train)
    np.save('{}y_val.npy'.format(base_dir), y_val)
    np.save('{}y_test.npy'.format(base_dir), y_test)

    return x_im_train,x_im_val,x_im_test,y_train,y_val,y_test


def load_data(config):

    if not os.path.exists(base_dir):
        print("Transform data first")
        return prepare_data(config)
        
    x_train = np.load(base_dir + 'x_im_train.npy')
    x_val = np.load(base_dir + 'x_im_val.npy')
    x_test = np.load(base_dir + 'x_im_test.npy')

    y_train = np.load(base_dir + 'y_train.npy')
    y_val = np.load(base_dir + 'y_val.npy')
    y_test = np.load(base_dir + 'y_test.npy')

    return x_train,x_val,x_test, y_train,y_val,y_test



def train(config):
    img_width, img_height = config['img_width'], config['img_height']
    batch_size = config['batch_size']
    cw = config['cw']
    model_name = config['model_name']

    model = build_model(config, img_width, img_height)

    x_train,x_val,x_test, y_train, y_val, y_test = load_data(config)

    #Normalizing input
    #Rescale to [-1,1]
    x_train =(x_train)/255.*2-1
    x_val = (x_val)/255.*2-1
    x_test = (x_test)/255.*2-1

    #Data augmentation
    train_gen = ImageDataGenerator(
            featurewise_center = False,
            samplewise_center = False,
            featurewise_std_normalization = False,
            samplewise_std_normalization = False,
            rotation_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1
    )

    train_gen.fit(x_train)
    tg_flow = train_gen.flow(x_train,y_train, batch_size=batch_size)

    #Balance classes
    if cw == 1:
        print("Balanced weighting")
        class_weights = class_weight.compute_class_weight('balanced', np.unique(np.argmax(y_train,1)),np.argmax(y_train,1))
        print(class_weights)
        d_class_weights = dict(enumerate(class_weights))
    elif cw == 0:
        print("Equal weighting")
        class_weights = [1,1,1,1,1,1,1]
        d_class_weights = dict(enumerate(class_weights))
    elif cw == 2:
        print("Custom weighting")
        class_weights = [2,2,3,1,3,0.7, 2]
        d_class_weights = dict(enumerate(class_weights))

    #Callbacks
    save_dir = 'checkpoints/{}'.format(model_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath= os.path.join(save_dir,model_name + "-{epoch:02d}-{val_acc:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor = 'val_acc', min_delta = 0, patience = 7)
    rlr = ReduceLROnPlateau(monitor = 'val_acc',
            patience = 3,
 )

    
    callback_list = [rlr, checkpoint, early_stopping]
    epochs = config['epochs']

    history = model.fit_generator(tg_flow,epochs = epochs,
                              verbose = 1, validation_data = (x_val, y_val),
                              steps_per_epoch=x_train.shape[0] // batch_size,
                              callbacks = callback_list,
                              class_weight = d_class_weights
                            
    )

    base_dir = os.path.join('saved_models/',model_name)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    scores = model.evaluate(x_test,y_test)

    
    model.save(base_dir + '/model.h5')
    dd.io.save(base_dir + '/model_scores.h5', scores)
    dd.io.save(base_dir + '/model_val_history' + '.h5', history.history)

import argparse
import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-net", "--net", type=str, required = False)
    parser.add_argument("-e", "--epochs", type = int,  required = False)
    parser.add_argument("-n", "--num_trainable", type = int,  required = False)
    parser.add_argument("-cw", "--cw", type = int, required = False)
    parser.add_argument("-t", "--TEST", type = str, required =False)

    parser.add_argument('-hi', '--img_height', type = int, required = False)
    parser.add_argument('-wi', '--img_width', type = int, required = False)
    args = vars(parser.parse_args())

    now = datetime.datetime.now()
    
    batch_size = 16
    model_name = '{}_{}_{}_{}_T_{}-{}'.format(args['net'],args['cw'],args['num_trainable'], batch_size,now.hour, now.minute)
    args['model_name'] = model_name
    args['batch_size'] = batch_size
    args['base_dir'] = base_dir
    if args['img_height'] is not None and args['img_width'] is not None:
        global base_dir
        img_height = args['img_height']
        img_width = args['img_width']
        base_dir = 'prepared_data_{}_{}/'.format(img_height, img_width)
        print(base_dir)
        args['base_dir'] = base_dir
    else:
        args['img_width'] = 226
        args['img_height'] = 226

    
    train(args)

    print("Done")

if __name__ == '__main__':
    main()
