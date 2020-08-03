# [U-Net](https://arxiv.org/abs/1505.04597) Keras Zoo

- [U-Net Keras Zoo](#u-net-keras-zoo)
  - [Network Types](#network-types)
    - [U-Net](#u-net)
    - [U-Xception](#u-xception)
    - [URes-Xception](#ures-xception)
    - [USpp-Xception](#uspp-xception)
    - [U-ResNet50V2](#u-resnet50v2)
    - [URes-ResNet50V2](#ures-resnet50v2)
    - [USpp-ResNet50V2](#uspp-resnet50v2)
    - [DeepRes-U-Net](#deepres-u-net)
    - [D-UNet](#d-unet)
  - [Network Inputs](#network-inputs)

## Network Types

This repository contains some implementations of mine of the well-known U-Net neural network. Most of the networks in the Zoo are based on the same architecture. Here is a scheme of it:

![net_architecture](https://i.ibb.co/bKZK4nH/net-ushape-w-legend-18.png)

The skeleton is basically a U-Net that uses a pre-trained model on the encoder part. What varies between the different networks are such pre-trained models and the encoder blocks used to process the intermediate outputs.

I use two different types of blocks: [Spatial Pyramid Pooling blocks](https://arxiv.org/abs/1406.4729) and [Residual blocks](https://arxiv.org/abs/1512.03385). Below is a sketch of them. Note that in the networks, the decoder blocks are always residual blocks, because they have shown better results when decoding images.

![blocks](https://i.ibb.co/TLY2xzw/blocks-legend-v4.png)

Enough with the high-level descriptions. Let's dig into the single networks:

### [U-Net](/nets/unet.py)
It's a plain U-Net. No pretrained networks is used as encoder and no block is used to process the intermediate outputs.

### [U-Xception](/nets/u_xception.py)
It's a U-Net that uses an Xception Net pretrained on the 'imagenet' dataset as encoder. The intermediate outputs of the Xception Net are taken as they are and fed to the decoders.

### [URes-Xception](/nets/ures_xception.py)
It's another U-Net that uses an Xception Net pretrained on the 'imagenet' dataset as encoder, but this time the intermediate outputs of the Xception Net are processed by two residual blocks before being fed to the decoders.

### [USpp-Xception](/nets/uspp_xception.py)
It's the same architecture of the network above, but instead of being processed by two residual blocks, the Xception outputs are refined by Spatial Pyramid blocks.

### [U-ResNet50V2](/nets/u_resnet50v2.py)
It's a U-Net that uses a ResNet50V2 pretrained on the 'imagenet' dataset as encoder. The intermediate outputs of the ResNet50V2 are vanilla fed to the decoders.

### [URes-ResNet50V2](/nets/ures_resnet50v2.py)
It's a U-Net that uses a pretrained ResNet50V2 as encoder. Like in the URes-Xception Net, the intermediate outputs of the pretrained net are processed by two residual blocks before being fed to the decoders.

### [USpp-ResNet50V2](/nets/uspp_resnet50v2.py)
It's the same architecture of the network above, but instead of being processed by two residual blocks, the ResNetV2 outputs are refined by Spatial Pyramid blocks.

### [DeepRes-U-Net](/nets/deepresunet.py)
It's a deep U-Net that does not use any pretrained net as encoder, and processes every intermediate output with residual blocks.

### [D-UNet](/nets/dunet.py)
It's a dimension fusion U-Net, that process the input both in 4D and 3D, before mixing all together. It's the implementation of [this paper](https://arxiv.org/abs/1908.05104).

## Network Inputs

All the networks are implemented with the same input size of ```400, 400, 3```. If you need to work with different image sizes, you need to change the input size of the networks. Moreover, depending on which size you are using, you might need to change some intermediate layer: if you are still beginning to learn Keras and you need some help with it, feel free to contact me, via [github](https://github.com/daniCh8) or [mail](mailto:daniele.chiappalupi@gmail.com); I'll be happy to help.