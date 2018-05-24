# ZF_UNET_224 Pretrained Model
Modification of convolutional neural net "UNET" for image segmentation in Keras framework

## Requirements

Python 3.*, Keras 2.1, Tensorflow 1.4

## Usage

```python
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef
from keras.optimizers import Adam

model = ZF_UNET_224(weights='generator')
optim = Adam()
model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

model.fit(...)
```

## Notes

- "ZF_UNET_224" Model based on UNET code from following paper: https://arxiv.org/abs/1505.04597
- This model used to get 2nd place in DSTL competition: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection
- For training used DICE coefficient: https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
- Input shape for model is 224x224 (the same as for other popular CNNs like VGG or ResNet)
- It has 3 input channels (to process standard RGB (BGR) images). You can change it with variable "INPUT_CHANNELS"
- In most cases model ZF_UNET_224 is ok to be used without pretrained weights.
- This code should work fine on both Theano and Tensorflow backends. Code prepared for Keras 2.1, if you need code for Keras 1.2 then use this [link](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/tree/b68bbf3a8af4b732a68cf693fcaa59ae19a0e5e5):

## Pretrained weights

Download: [Weights for Tensorflow backend ~123 MB (Keras 2.1, Dice coef: 0.998)](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/releases/download/v1.0/zf_unet_224.h5)

Weights were obtained with random image generator (generator code available here: train_infinite_generator.py). See example of images from generator below.

![Example of images from generator](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/img/ZF_UNET_Generator_Images_Example.png)

Dice coefficient for pretrained weights: **~0.998**. See history of learning below:

![Log of dice coefficient during training process](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/img/Dice_log.png)
