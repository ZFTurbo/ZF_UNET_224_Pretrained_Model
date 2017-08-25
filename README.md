# ZF_UNET_224 Pretrained Model
Modification of convolutional neural net "UNET" for image segmentation in Keras framework

## Requirements

Python 3.*, Keras 1.2, Theano 0.9

## Usage

```python
from zf_unet_224_model import ZF_UNET_224, dice_coef_loss, dice_coef
from keras.optimizers import Adam

model = ZF_UNET_224()
model.load_weights("zf_unet_224.h5")
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
- This code should work fine on both Theano and Tensorflow backends. It will work on Keras 2.0 as well with some warnings about function old namings and parameters.

## Pretrained weights

Download: [Weights for Theano backend ~123 MB](https://mega.nz/#!eAY2WAJS!zsb9rq20gjaSWJECu6tGdTN9tG6ZzQk0KQvB8iG2sL4)

Weights were obtained with random image generator (generator code available here: train_infinite_generator.py). See example of images from generator below.

![Example of images from generator](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/img/ZF_UNET_Generator_Images_Example.png)

Dice coefficient for pretrained weights: **~0.999**. See history of learning below:

![Log of dice coefficient during training process](https://github.com/ZFTurbo/ZF_UNET_224_Pretrained_Model/blob/master/img/Dice_log.png)

