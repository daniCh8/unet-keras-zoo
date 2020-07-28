import keras
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, MaxPooling2D, Dropout, Concatenate, concatenate, ZeroPadding2D, Conv2DTranspose
from keras.applications import Xception
from keras.models import Model

def spatial_pyramid_block(x, filters, d_a, d_b, activation=True, pad=0):
    f = int(filters/4)

    x_1 = Conv2D(f, (3, 3), padding='same') (x)
    x_1 = BatchNormalization() (x_1)
    x_2 = Conv2D(f, (3, 3), dilation_rate=(d_a, d_a), padding='same') (x)
    x_2 = BatchNormalization() (x_2)
    x_3 = Conv2D(f, (3, 3), dilation_rate=(d_b, d_b), padding='same') (x)
    x_3 = BatchNormalization() (x_3)
    x_4_down = MaxPooling2D(pool_size=(4, 4)) (x)
    x_4_up = Conv2DTranspose(f, (3, 3), strides=(4, 4)) (x_4_down)
    if pad != 0:
        x_4_up = ZeroPadding2D(padding=((pad, 0), (pad, 0))) (x_4_up)
    x_4_up = BatchNormalization() (x_4_up)
    
    cc = Concatenate() ([x_1, x_2, x_3, x_4_up])
    cc = LeakyReLU(alpha=.1) (cc)
    cc_conv = Conv2D(filters, (3, 3), padding='same') (cc)
    cc_conv = BatchNormalization() (cc_conv)
    if activation:
        cc_conv = LeakyReLU(alpha=.1) (cc_conv)
    return cc_conv

def conv_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x

def res_block(x, filters):
    x_1 = LeakyReLU(alpha=.1) (x)
    x_1 = BatchNormalization() (x_1)
    x_2 = BatchNormalization() (x)
    x_1 = conv_block(x_1, filters, (3, 3))
    x_1 = conv_block(x_1, filters, (3, 3), activation=False)
    x = Add() ([x_1, x_2])
    return x

def decoder_block(x, filters, kernel_size=(3, 3), activation=True):
    x = Conv2D(filters, kernel_size, padding='same') (x)
    x = res_block(x, filters)
    x = res_block(x, filters)
    if activation == True:
        x = LeakyReLU(alpha=.1) (x)
    return x

def create_model():

    backbone = Xception(input_shape=(400, 400, 3),weights='imagenet',include_top=False)
    input_ = backbone.input

    conv4 = backbone.layers[121].output #(None, 25, 25, 1024)
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4) #(None, 12, 12, 1024)
    pool4 = Dropout(.5)(pool4)
    
    decoder4 = decoder_block(pool4, 512, (3, 3)) #(None, 12, 12, 512)
    deconv4 = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="valid")(decoder4) #(None, 25, 25, 256)

    conv4 = spatial_pyramid_block(conv4, 256, 2, 3, activation=False, pad=1) #(None, 25, 25, 256)
    uconv4 = concatenate([deconv4, conv4]) #(None, 25, 25, 512)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)
    uconv4 = Dropout(.5)(uconv4)
    
    decoder3 = decoder_block(uconv4, 256, (3, 3)) #(None, 25, 25, 256)
    deconv3 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(decoder3) #(None, 50, 50, 128)
    conv3 = backbone.layers[31].output #(None, 50, 50, 728)
    conv3 = LeakyReLU(alpha=0.1)(conv3)
    conv3 = spatial_pyramid_block(conv3, 128, 2, 3, activation=False, pad=2) #(None, 50, 50, 128)

    uconv3 = concatenate([deconv3, conv3]) #(None, 50, 50, 256)
    uconv3 = Dropout(.5)(uconv3)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)
    
    decoder2 = decoder_block(uconv3, 128, (3, 3)) #(None, 50, 50, 128)
    deconv2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(decoder2) #(None, 100, 100, 64)
    conv2 = backbone.layers[21].output #(None, 99, 99, 256)
    conv2 = LeakyReLU(alpha=0.1)(conv2)
    conv2 = ZeroPadding2D(((1,0),(1,0)))(conv2) #(None, 100, 100, 256)
    conv2 = spatial_pyramid_block(conv2, 64, 2, 3, activation=False) #(None, 100, 100, 64)

    uconv2 = concatenate([deconv2, conv2]) #(None, 100, 100, 128)
    uconv2 = Dropout(.1)(uconv2)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    decoder1 = decoder_block(uconv2, 64, (3, 3)) #(None, 100, 100, 64)
    deconv1 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(decoder1) #(None, 200, 200, 32)
    conv1 = backbone.layers[11].output #(None, 197, 197, 128)
    conv1 = LeakyReLU(alpha=0.1)(conv1)
    conv1 = ZeroPadding2D(((3,0),(3,0)))(conv1) #(None, 200, 200, 128)
    conv1 = spatial_pyramid_block(conv1, 32, 2, 3, activation=False) #(None, 200, 200, 32)

    uconv1 = concatenate([deconv1, conv1]) #(None, 200, 200, 64)
    uconv1 = Dropout(.1)(uconv1)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    decoder0 = decoder_block(uconv1, 32, (3, 3)) #(None, 200, 200, 32)
    deconv0 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding="same")(decoder0) #(None, 400, 400, 16)
    conv0 = spatial_pyramid_block(input_, 16, 2, 3, activation=False) #(None, 400, 400, 16)

    uconv0 = concatenate([deconv0, conv0]) #(None, 400, 400, 32)
    uconv0 = Dropout(.4)(uconv0)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    final_decoder = decoder_block(uconv0, 16, (3, 3)) #(None, 400, 400, 16)
    final_decoder = Dropout(.25)(final_decoder)
    output_ = Conv2D(1, (1,1), padding="same", activation="sigmoid")(final_decoder) #(None, 400, 400, 1)
    
    model = Model(inputs=input_, outputs=output_)
    model.name = 'uspp_xception'

    return model