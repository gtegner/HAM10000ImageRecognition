from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, AveragePooling2D, Flatten, GlobalMaxPool2D, Input, BatchNormalization
from keras.layers import Activation
from keras.optimizers import Adam, SGD
from keras.applications import VGG16, DenseNet121

from keras.models import Sequential

from keras.models import Model
from keras.layers import Conv2D, Dense, Dropout, MaxPool2D, AveragePooling2D, Flatten, GlobalMaxPool2D, Input, BatchNormalization
from keras.layers import Activation, Add
from keras.optimizers import Adam, SGD
from keras.applications import VGG16, DenseNet121

from keras.models import Sequential

def Model_1(img_height, img_width):
    
    inp = Input(shape = (img_height,img_width,3))
    
    x = inp

    x = Conv2D(32, (5,5), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    
    x_res = x
    
    x = Conv2D(32, (3,3), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(32, (3,3), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x_res, x])
    
    x = MaxPool2D(pool_size = 2)(x)
    x = Dropout(0.25)(x)    

    x_res = Conv2D(64, (1,1))(x)
    x = Conv2D(64, (3,3), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64, (3,3), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, x_res])
    
    x = MaxPool2D(pool_size = 2)(x)
    x = Dropout(0.4)(x)
    
    x_res = Conv2D(96, (1,1))(x)
    x = Conv2D(96, (3,3), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    

    
    x = Add()([x, x_res])
    x = MaxPool2D(pool_size = 2)(x)
    x = Dropout(0.4)(x)
    
    x_res = Conv2D(128, (1,1))(x)
    x = Conv2D(128, (3,3), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    
    x = Add()([x, x_res])
    x = MaxPool2D(pool_size = 2)(x)
    
    x = Flatten()(x)

    x = Dense(128, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation = 'softmax')(x)

    model = Model(inputs = inp, outputs = x)
    optimizer = Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    return model 

def Model_2(img_height, img_width):
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding = 'same', activation = 'relu', input_shape = (img_height, img_width, 3)))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'same'))
    
    model.add(MaxPool2D(pool_size = (2, 2)))
    
    model.add(Dropout(0.25))


    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.4))
    

    model.add(Conv2D(96, (3, 3), activation='relu',padding = 'Same'))
    model.add(Conv2D(96, (3, 3), activation='relu',padding = 'Same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def Model_3(img_height, img_width):
    inp = Input(shape = (img_height,img_width,3))
    x = inp

    x = Conv2D(32, (5,5), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3,3), strides = 1, padding = 'same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPool2D(pool_size = 2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3,3), padding = 'same')(x)
    x = Activation('relu')(x)    
    x = Conv2D(64, (3,3), padding = 'same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPool2D(pool_size = 2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(96, (3,3), strides = 1, padding = 'same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(96, (3,3), strides = 1, padding = 'same')(x)
    #x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    x = MaxPool2D(pool_size = 2)(x)
    x = Dropout(0.4)(x)

    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation = 'softmax')(x)

    model = Model(inputs = inp, outputs = x)
    optimizer = Adam(lr = 0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model    




def Model_4(input_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',padding = 'same',input_shape=input_shape))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu',padding = 'same'))
    model.add(MaxPool2D(pool_size = 2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))
    model.add(Conv2D(64, (3, 3), activation='relu',padding = 'same'))

    model.add(MaxPool2D(pool_size=2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

def smaller_model(img_height, img_width):
    model = Sequential()
    model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu', padding = 'same', input_shape = (img_height, img_width, 3)))
    model.add(Conv2D(16, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu', padding = 'same'))
    model.add(MaxPool2D(pool_size = (2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64, activation = 'relu'))
    model.add(Dropout(0.4))
    model.add(Dense(7, activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model



def build_model(config, img_width, img_height):
    net_type = config['net']

    if net_type == 'VGG':
        net = VGG16(include_top=False,weights='imagenet', input_shape = (img_height,img_width, 3))
    elif net_type == 'CUSTOM':
        return Model_1(img_height, img_width)  
    else:
        print("invalid input")
        return -1
    


    #Transfer learning from vgg net    
    for layer in net.layers:
        layer.trainable = False

    x = net.get_layer('block1_pool').output
    x = Conv2D(64, (3,3), activation = 'relu')(x)
    x = Conv2D(64, (3,3), activation = 'relu')(x)
    x = MaxPool2D(pool_size = (2,2))(x)
    x = Conv2D(128, (3,3), activation = 'relu')(x)
    x = Conv2D(128, (3,3), activation = 'relu')(x)
    x = MaxPool2D(pool_size = (2,2))(x)
    x = Flatten()(x)

    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(7, activation = 'softmax')(x)
    
    optimizer = Adam(lr = 0.001)
    model = Model(inputs = net.input, outputs = x)
    model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    
    return model
