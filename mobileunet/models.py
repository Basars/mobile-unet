import tensorflow as tf
from tensorflow.keras import layers, applications, Model, Sequential


ENCODER_BLOCK_NAMES = [
    'block_1_expand_relu',
    'block_3_expand_relu',
    'block_6_expand_relu',
    'block_13_expand_relu',
    'block_16_project'  # bottleneck
]

DECODER_BLOCK_FILTERS = [512, 256, 128, 64]


def decoder_block(filters, kernel_size, name='decoder_block'):
    initializer = tf.random_normal_initializer(0., 0.02)
    model = Sequential(name=name, layers=[
        layers.Conv2DTranspose(filters, kernel_size,
                               strides=2, padding='same',
                               kernel_initializer=initializer,
                               use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU()
    ])
    return model


def encoder(input_shape, name='encoder'):
    base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False)
    skips = [base_model.get_layer(name).output for name in ENCODER_BLOCK_NAMES]

    encoder_model = Model(inputs=base_model.input, outputs=skips, name=name)
    encoder_model.trainable = False
    return encoder_model


def build_mobile_unet(input_shape=(224, 224, 3), output_channels=1, name='mobile_unet'):
    inputs = layers.Input(shape=input_shape)

    decoder_blocks = [decoder_block(filters, 3, name='decoder_block_{}'.format(block_id))
                      for block_id, filters in enumerate(DECODER_BLOCK_FILTERS)]
    encoder_model = encoder(input_shape)

    x = inputs
    skips = encoder_model(x)
    x = skips[-1]  # bottleneck outputs
    skips = reversed(skips[:-1])

    for decoder, skip in zip(decoder_blocks, skips):
        x = decoder(x)
        x = layers.Concatenate()([x, skip])

    x = layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')(x)
    return Model(inputs=inputs, outputs=x, name=name)


MobileUNet = build_mobile_unet
