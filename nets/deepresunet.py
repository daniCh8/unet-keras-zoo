import keras
from keras.layers import Conv2D, Dropout, Concatenate, Input, BatchNormalization, LeakyReLU, Add, Conv2DTranspose, MaxPooling2D
from keras.models import Model

def conv_block(x, filters, strides=(1, 1)):
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    return x

def res_block(x, filters, downsample=True, activation=False):
    if downsample:
        strides = (2, 2)
    else:
        strides = (1, 1)

    x_1 = conv_block(x, filters, strides)
    x_1 = conv_block(x_1, filters)
    
    x_2 = conv_block(x, filters, strides)

    x = Add()([x_1, x_2])
    if activation:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=.1)(x)
    x = Dropout(.1)(x)
    return x

def decoder_block(x, shortcut, filters):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(x)
    cc = Concatenate()([x, shortcut])
    cc = Dropout(.2)(cc)
    out = res_block(cc, filters, downsample=False, activation=True)
    return out

def create_model():
    input_ = Input((400, 400, 3))
    conv1 = Conv2D(64, (3, 3), padding='same')(input_)
    encoder1 = conv_block(conv1, 64)
    encoder1 = Add() ([encoder1, conv1]) #400, 400, 64

    encoder2 = res_block(encoder1, 128) #200, 200, 128
    encoder3 = res_block(encoder2, 256) #100, 100, 256
    encoder4 = res_block(encoder3, 512) #50, 50, 512
    encoder5 = res_block(encoder4, 768, activation=True) #25, 25, 768

    decoder4 = decoder_block(encoder5, encoder4, 256) #50, 50, 512
    decoder3 = decoder_block(decoder4, encoder3, 128) #100, 100, 256
    decoder2 = decoder_block(decoder3, encoder2, 64) #200, 200, 128
    decoder1 = decoder_block(decoder2, encoder1, 32) #400, 400, 64

    output_ = Dropout(.4)(decoder1)
    output_ = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(output_) #400, 400, 1

    model = Model(input_, output_)
    model.name = 'deep-res-unet'

    return model
