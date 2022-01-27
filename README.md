# Mobile-UNet

An ML model with U-shaped architecture with MobileNetV2 based encoders

#### Install
```bash
pip install --upgrade git+https://github.com/Basars/mobile-unet.git
```

#### Usage:
```python
from mobileunet import MobileUNet

model = MobileUNet(input_shape=(224, 224, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(...)
```