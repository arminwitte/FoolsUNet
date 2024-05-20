import foolsunet.layers as fl
import tensorflow as tf
from tensorflow.keras import layers


def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    result.add(layers.LeakyReLU())

    return result


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(
            filters,
            size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    result.add(layers.BatchNormalization())

    if apply_dropout:
        result.add(layers.Dropout(0.5))

    result.add(layers.ReLU())

    return result


def downsample1(filters, size, apply_batchnorm=True):

    result = tf.keras.Sequential(
        [
            fl.InverseResidualBlock(filters * 2 // 3),
            fl.InverseResidualBlock(filters * 2 // 3),
            fl.InverseResidualBlock(filters, strides=2),
        ]
    )
    return result


def upsample1(filters, size, apply_dropout=False):

    result = tf.keras.Sequential(
        [
            fl.InverseResidualBlock(filters * 3 // 2),
            fl.InverseResidualBlock(filters * 3 // 2),
            # layers.UpSampling2D(size=2),
            layers.Conv2DTranspose(
                filters, size, strides=2, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]
    )

    return result


def foolsunet(num_transformers=0):
    inputs = layers.Input(shape=[256, 256, 3])
    # initializer = tf.random_normal_initializer(0., 0.02)
    # first = layers.Conv2D(64, 3,
    #                                      strides=1,
    #                                      padding='same',
    #                                      kernel_initializer=initializer,
    #                                      activation='relu6')  # (batch_size, 256, 256, 3)

    down_stack = [
        # InverseResidualBlock(24, strides=2),
        downsample(64, 3, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        # InverseResidualBlock(32, strides=2),
        downsample(128, 3),  # (batch_size, 64, 64, 128)
        downsample1(64, 3),  # (batch_size, 32, 32, 256)
        downsample1(96, 3),  # (batch_size, 16, 16, 512)
        downsample1(128, 3),  # (batch_size, 8, 8, 512)
        # downsample1(128, 4),  # (batch_size, 4, 4, 512)
        # downsample1(192, 4),  # (batch_size, 2, 2, 512)
        # downsample1(512, 4),  # (batch_size, 1, 1, 512)
    ]

    transformers = []
    if num_transformers > 0:
        transformers.append(
            fl.PatchAndEncode(num_patches=64, patch_size=4, projection_dim=128)
        )

    for _ in range(num_transformers):
        transformers.append(fl.TransformerBlock(num_heads=12, projection_dim=128))

    if num_transformers > 0:
        transformers.append(fl.MakeShape([None, 32, 32, 48]))

    up_stack = [
        # upsample1(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        # upsample1(128, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        # upsample1(96, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        upsample1(96, 3),  # (batch_size, 16, 16, 1024)
        upsample1(64, 3),  # (batch_size, 32, 32, 512)
        upsample(128, 3),  # (batch_size, 64, 64, 256)
        upsample(64, 3),  # (batch_size, 128, 128, 128)
    ]

    attention_stack = [fl.Attention() for _ in range(len(down_stack) - 1)]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = layers.Conv2DTranspose(
        3, 3, strides=2, padding="same", kernel_initializer=initializer
    )  # (batch_size, 256, 256, 3)

    x = inputs
    # x = first(x)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # bottleneck
    for t in transformers:
        x = t(x)

    # Upsampling and establishing the skip connections
    for up, skip, attention in zip(up_stack, skips, attention_stack):
        x = up(x)
        # skip = attention([x, skip])
        x = layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
