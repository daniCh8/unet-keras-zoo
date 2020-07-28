import keras
from keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Add, MaxPooling2D, Dropout, Concatenate, ZeroPadding2D, Conv2DTranspose
from keras.applications import ResNet50V2
from keras.models import Model

def spatial_pyramid_block(x, filters, d_a, d_b, activation=False, pad=0):
    f = int(filters/4)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)

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
    if activation:
        cc_conv = BatchNormalization() (cc_conv)
        cc_conv = LeakyReLU(alpha=.1) (cc_conv)
    return cc_conv

def conv_block(x, filters, strides=(1, 1)):
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    x = Conv2D(filters, (3, 3), strides=strides, padding='same')(x)
    return x

def res_block(x, filters, activation=False):
    x_1 = conv_block(x, filters)
    x_1 = conv_block(x_1, filters)
    x_2 = conv_block(x, filters)

    x = Add() ([x_1, x_2])

    if activation:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=.1)(x)

    return x

def create_model():
    input_ = Input((400, 400, 3))
    backbone = ResNet50V2(input_tensor=input_, weights='imagenet', include_top=False)
    
    layer6 = backbone.get_layer('post_bn').output #13, 13, 2048
    layer6 = BatchNormalization()(layer6)
    layer6 = LeakyReLU(alpha=.1)(layer6)

    decoder5 = Conv2DTranspose(256, (3, 3), strides=(2,2))(layer6) #26, 26, 256
    decoder5 = BatchNormalization()(decoder5)
    decoder5 = LeakyReLU(alpha=.1)(decoder5)
    decoder5 = Conv2D(256, (3, 3))(decoder5) #25, 25, 256
    layer5 = backbone.get_layer('conv4_block5_out').output #25, 25, 1024
    layer5 = spatial_pyramid_block(layer5, 256, 2, 3, pad=1) #25, 25, 256
    decoder5 = Concatenate()([decoder5, layer5]) #25, 25, 512
    decoder5 = res_block(decoder5, 256, activation=True) #25, 25, 256

    decoder4 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(decoder5) #50, 50, 128
    layer4 = backbone.get_layer('conv3_block4_1_conv').output #50, 50, 128
    layer4 = spatial_pyramid_block(layer4, 128, 2, 3, pad=2) #50, 50, 128
    decoder4 = Concatenate()([decoder4, layer4]) #50, 50, 256
    decoder4 = Dropout(.2)(decoder4)
    decoder4 = res_block(decoder4, 128, activation=True) #50, 50, 128

    decoder3 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(decoder4) #100, 100, 64
    layer3 = backbone.get_layer('conv2_block3_1_conv').output #100, 100, 64
    layer3 = spatial_pyramid_block(layer3, 64, 2, 3) #100, 100, 64
    decoder3 = Concatenate()([decoder3, layer3]) #100, 100, 128
    decoder3 = Dropout(.2)(decoder3)
    decoder3 = res_block(decoder3, 64, activation=True) #100, 100, 64

    decoder2 = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(decoder3) #200, 200, 32
    layer2 = backbone.get_layer('conv1_conv').output #200, 200, 64
    layer2 = spatial_pyramid_block(layer2, 32, 2, 3) #200, 200, 32
    decoder2 = Concatenate()([decoder2, layer2]) #200, 200, 64
    decoder2 = Dropout(.2)(decoder2)
    decoder2 = res_block(decoder2, 32, activation=True) #200, 200, 32

    decoder1 = Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same')(decoder2) #400, 400, 16
    layer1 = input_ #400, 400, 3
    layer1 = spatial_pyramid_block(layer1, 16, 2, 3) #400, 400, 16
    decoder1 = Concatenate()([decoder1, layer1]) #400, 400, 32
    decoder1 = Dropout(.2)(decoder1)
    decoder1 = res_block(decoder1, 16, activation=True) #400, 400, 16
    decoder1 = Dropout(.3)(decoder1)

    output_ = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(decoder1) #400, 400, 1

    model = Model(input_, output_)
    model.name = 'uspp_resnet50v2'

    return model