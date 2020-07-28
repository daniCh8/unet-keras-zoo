import keras
from keras.layers import Input, Conv2D, BatchNormalization, Dropout, Concatenate, MaxPooling2D, Conv3D, Conv2DTranspose, Reshape, Dense, LeakyReLU, Add, MaxPooling3D, Multiply, GlobalAveragePooling2D
from keras.models import Model


def conv3d_block(x, filters):
    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)

    x = Conv3D(filters, (3, 3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    return x

def conv2d_block(x, filters):
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)

    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=.1)(x)
    return x

def squeeze_excitation_block(in_, c, r):
    x = GlobalAveragePooling2D()(in_)
    x = Dense(c//r, activation='relu')(x)
    x = Dense(c, activation='sigmoid')(x)
    return Multiply()([in_, x])
 
def dimension_fusion_box(in_4d, in_3d, shape, depth):
    x_4d = Conv3D(1, (1, 1, 1), padding='same')(in_4d)
    x_4d = Reshape(target_shape=(shape[0], shape[1], depth))(x_4d)
    x_4d = Conv2D(shape[2], (3, 3), padding='same')(x_4d)

    x_4d = squeeze_excitation_block(x_4d, shape[2], 16)
    x_3d = squeeze_excitation_block(in_3d, shape[2], 16)

    x = Add()([x_3d, x_4d])
    return x

def decoder_block(in_a, in_b, filters, padding='same', dropout=.1):
    x = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding=padding)(in_a)
    x = Concatenate()([x, in_b])
    x = conv2d_block(x, filters)
    x = Dropout(dropout)(x)
    return x

def create_model():
    input_ = Input(shape=(400, 400, 3))
    input_4d = Reshape(target_shape=(400, 400, 3, 1))(input_)

    track_4d_1 = conv3d_block(input_4d, 32) #400, 400, 3, 32
    track_4d_1_up = MaxPooling3D(pool_size=(2, 2, 1))(track_4d_1) #200, 200, 3, 32

    track_4d_2 = conv3d_block(track_4d_1_up, 64) #200, 200, 3, 64
    track_4d_2_up = MaxPooling3D()(track_4d_2) #100, 100, 1, 64

    track_4d_3 = conv3d_block(track_4d_2_up, 128) #100, 100, 1, 128
    track_4d_3_up = MaxPooling3D(pool_size=(2, 2, 1))(track_4d_3) #50, 50, 1, 128 **new**

    track_4d_4 = conv3d_block(track_4d_3_up, 256) #50, 50, 1, 256 **new**

    track_3d_1 = conv2d_block(input_, 32) #400, 400, 32
    track_3d_1_up = MaxPooling2D()(track_3d_1) #200, 200, 32
    track_3d_2 = conv2d_block(track_3d_1_up, 64) #200, 200, 64
    track_3d_2 = Dropout(.1)(track_3d_2)

    track_3d_3 = dimension_fusion_box(track_4d_2, track_3d_2, [200, 200, 64], 3) #200, 200, 64
    track_3d_3_up = MaxPooling2D()(track_3d_3) #100, 100, 64
    track_3d_3_conv = conv2d_block(track_3d_3_up, 128) #100, 100, 128
    track_3d_3_conv = Dropout(.1)(track_3d_3_conv)

    track_3d_4 = dimension_fusion_box(track_4d_3, track_3d_3_conv, [100, 100, 128], 1) #100, 100, 128
    track_3d_4_up = MaxPooling2D()(track_3d_4) #50, 50, 128
    track_3d_4_conv = conv2d_block(track_3d_4_up, 256) #50, 50, 256
    track_3d_4_conv = Dropout(.1)(track_3d_4_conv)

    track_3d_5 = dimension_fusion_box(track_4d_4, track_3d_4_conv, [50, 50, 256], 1) #50 50, 256 **new**
    track_3d_5_up = MaxPooling2D()(track_3d_5) #25, 25, 256
    track_3d_5_conv = conv2d_block(track_3d_5_up, 512) #25, 25, 512
    track_3d_5_conv = Dropout(.1)(track_3d_5_conv)

    track_3d_6 = MaxPooling2D()(track_3d_5_conv) #12, 12, 512
    track_3d_6_conv = conv2d_block(track_3d_6, 768) #12, 12, 768
    track_3d_6_conv = Dropout(.1)(track_3d_6_conv)

    track_3d_7 = decoder_block(track_3d_6_conv, track_3d_5_conv, 512, padding='valid') #25, 25, 512
    track_3d_8 = decoder_block(track_3d_7, track_3d_5, 256) #50, 50, 256
    track_3d_9 = decoder_block(track_3d_8, track_3d_4, 128) #100, 100, 128
    track_3d_10 = decoder_block(track_3d_9, track_3d_3, 64) #200, 200, 64
    track_3d_11 = decoder_block(track_3d_10, track_3d_1, 32, dropout=.3) #400, 400, 32

    output_ = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(track_3d_11) #400, 400, 1

    model = Model(input_, output_)
    model.name = 'd-unet'
    
    return model
