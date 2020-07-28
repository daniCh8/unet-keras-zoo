import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, concatenate, Conv2DTranspose
from keras.models import Model

def conv_block(filters, size, dropout, x):
    conv = Conv2D(filters, size, activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (x)
    conv = BatchNormalization() (conv)
    conv = Dropout(dropout) (conv)
    conv = Conv2D(filters, size, activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (conv)
    conv = BatchNormalization() (conv)
    return conv

def conv_transpose_block(filters, dropout, x, to_concatenate, axis= -1):
    upsample = Conv2DTranspose(filters, (2, 2), strides= (2, 2), padding= 'same') (x)
    upsample = concatenate([upsample, to_concatenate], axis= axis)

    conv = Conv2D(filters, (3, 3), activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (upsample)
    conv = BatchNormalization() (conv)
    conv = Dropout(dropout) (conv)
    conv = Conv2D(filters, (3, 3), activation= 'elu', kernel_initializer= 'he_normal', padding= 'same') (conv)
    conv = BatchNormalization() (conv)
    return conv

def create_model():
    input = Input((400, 400, 3))

    conv1 = conv_block(16, (3, 3), 0.1, input)
    layer_1 = MaxPooling2D((2, 2)) (conv1)

    conv2 = conv_block(32, (3, 3), 0.1, layer_1)
    layer_2 = MaxPooling2D((2, 2)) (conv2)

    conv3 = conv_block(64, (3, 3), 0.2, layer_2)
    layer_3 = MaxPooling2D((2, 2)) (conv3)

    conv4 = conv_block(128, (3, 3), 0.2, layer_3)
    layer_4 = MaxPooling2D((2, 2)) (conv4)

    conv5 = conv_block(256, (3, 3), 0.3, layer_4)

    conv6 = conv_transpose_block(128, 0.2, conv5, conv4)
    conv7 = conv_transpose_block(64, 0.2, conv6, conv3)
    conv8 = conv_transpose_block(32, 0.2, conv7, conv2)
    conv9 = conv_transpose_block(32, 0.2, conv8, conv1, 3)

    output = Conv2D(1, (1, 1), activation='sigmoid') (conv9)

    model = Model(inputs= [input], outputs= [output])
    model.name = 'u_net'
    return model