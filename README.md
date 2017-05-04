# ZF_UNET_224_Pretrained_Model
Modification of convolutional neural net "UNET" for image segmentation in Keras framework

## Usage

```python
from a02_zf_unet_model import ZF_UNET_224, dice_coef_loss, dice_coef
from keras.optimizers import Adam

model = ZF_UNET_224()
optim = Adam()
model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

model.fit(...)
```
