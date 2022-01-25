import pytest
from mobileunet import MobileUNet


def test_mobile_unet_224x224_architecture():
    model = MobileUNet()

    assert tuple(model.input.shape) == (None, 224, 224, 3)
    assert tuple(model.output.shape) == (None, 224, 224, 1)


def test_mobile_unet_448x448_architecture():
    model = MobileUNet(input_shape=(448, 448, 3))

    assert tuple(model.input.shape) == (None, 448, 448, 3)
    assert tuple(model.output.shape) == (None, 448, 448, 1)


if __name__ == '__main__':
    pytest.main([__file__])
