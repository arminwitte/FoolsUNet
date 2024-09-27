import foolsunet.layers as fl
import tensorflow as tf
from tensorflow.keras import layers


def downsample(filters, size, apply_batchnorm=True, name=None):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential(name=name)
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


def upsample(filters, size, apply_dropout=False, name=None):
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential(name=name)
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


def downsample1(filters, size, channel_attention="", apply_batchnorm=True, name=None):

    result = tf.keras.Sequential(
        [
            # fl.InverseResidualBlock(filters * 2 // 3, channel_attention=channel_attention),
            # fl.InverseResidualBlock(filters * 2 // 3, channel_attention=channel_attention),
            # fl.InverseResidualBlock(filters, strides=2, channel_attention=channel_attention),
            
            fl.ASPPBlock(filters * 2 // 3, channel_attention=channel_attention),
            fl.ASPPBlock(filters * 2 // 3, channel_attention=channel_attention),
            fl.InverseResidualBlock(filters, strides=2, channel_attention=channel_attention),
        ],
    name=name)
    return result


def upsample1(filters, size, channel_attention="", apply_dropout=False, name=None):

    result = tf.keras.Sequential(
        [
            # fl.InverseResidualBlock(filters * 3 // 2, channel_attention=channel_attention),
            # fl.InverseResidualBlock(filters * 3 // 2, channel_attention=channel_attention),
            fl.ASPPBlock(filters * 3 // 2, channel_attention=channel_attention),
            fl.ASPPBlock(filters * 3 // 2, channel_attention=channel_attention),
            
            layers.Conv2DTranspose(
                filters, size, strides=2, padding="same", use_bias=False
            ),
            layers.BatchNormalization(),
            layers.ReLU(),
        ],
        name=name
    )

    return result


def foolsunet(num_transformers=0, channel_attention=""):
    inputs = layers.Input(shape=[256, 256, 3])
    # initializer = tf.random_normal_initializer(0., 0.02)
    # first = layers.Conv2D(64, 3,
    #                                      strides=1,
    #                                      padding='same',
    #                                      kernel_initializer=initializer,
    #                                      activation='relu6')  # (batch_size, 256, 256, 3)

    down_stack = [
        # InverseResidualBlock(24, strides=2),
        downsample(64, 3, apply_batchnorm=False, name="block_1_downsample"),  # (batch_size, 128, 128, 64)
        # InverseResidualBlock(32, strides=2),
        # downsample(128, 3, name="block_2_downsample"),  # (batch_size, 64, 64, 128)
        downsample1(32, 3, channel_attention=channel_attention, name="block_2_invres_downsample"),  # (batch_size, 64, 64, 128)
        downsample1(64, 3, channel_attention=channel_attention, name="block_3_invres_downsample"),  # (batch_size, 32, 32, 256)
        downsample1(96, 3, channel_attention=channel_attention, name="block_4_invres_downsample"),  # (batch_size, 16, 16, 512)
        downsample1(128, 3, channel_attention=channel_attention, name="block_5_invres_downsample"),  # (batch_size, 8, 8, 512)
        downsample1(192, 3, channel_attention=channel_attention, name="block_6_invres_downsample"),  # (batch_size, 4, 4, 512)
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
        upsample1(128, 3, channel_attention=channel_attention, apply_dropout=True , name="block_7_upsample"),  # (batch_size, 8, 8, 1024)
        upsample1(96, 3, channel_attention=channel_attention, name="block_8_invres_upsample"),  # (batch_size, 16, 16, 1024)
        upsample1(64, 3, channel_attention=channel_attention, name="block_9_invres_upsample"),  # (batch_size, 32, 32, 512)
        upsample(128, 3, name="block_10_upsample"),  # (batch_size, 64, 64, 256)
        upsample(64, 3, name="block_11_upsample"),  # (batch_size, 128, 128, 128)
    ]

    attention_stack = [fl.Attention() for _ in range(len(down_stack) - 1)]

    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = layers.Conv2DTranspose(
        3, 3, strides=2, padding="same", kernel_initializer=initializer
    , name="output")  # (batch_size, 256, 256, 3)

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




# ====================================================================================

def encoder(N=8, channel_attention="eca"):

    # Input layer (batch, 256, 256, 3)
    inputs = layers.Input(shape=[256, 256, 3], name="input")
    x = inputs

    # Initial conv block (batch, 256, 256, 3) -> (batch, 128, 128, 24)
    filters = 3 * N
    # x = fl.ASPPBlock2(filters, channel_attention=channel_attention, name="block_1_conv_0")(x)
    # x = fl.ASPPBlock2(filters, channel_attention=channel_attention, name="block_1_conv_1")(x)
    x = layers.Conv2D(
            filters,
            (3,3),
            strides=2,
            padding="same",
            # kernel_initializer=initializer,
            use_bias=False,
            name="stage_0_downsample",
        )(x)
    # x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    # stage 1 (batch, 128, 128, 24) -> (batch, 128, 128, 24)
    filters = 3 * N
    x = fl.FusedMBConvBlock(filters, expand_factor=1, channel_attention=channel_attention, name="stage_1_conv_0")(x)
    # x = fl.FusedMBConvBlock(filters, expand_factor=1, channel_attention=channel_attention, name="stage_1_conv_1")(x)


    # stage 2 (batch, 128, 128, 24) -> (batch, 64, 64, 48)
    filters = 6 * N
    # x = fl.FusedMBConvBlock(filters, expand_factor=4, channel_attention=channel_attention, name="stage_2_conv_0")(x)
    # x = fl.FusedMBConvBlock(filters, expand_factor=4, channel_attention=channel_attention, name="stage_2_conv_1")(x)
    # x = fl.FusedMBConvBlock(filters, expand_factor=4, channel_attention=channel_attention, name="stage_2_conv_2")(x)
    x = fl.FusedMBConvBlock(filters, expand_factor=4, strides=2, channel_attention=channel_attention, name="stage_2_conv_3")(x)


    # stage 3 (batch, 64, 64, 48) -> (batch, 32, 32, 64)
    filters = 8 * N
    # x = fl.FusedMBConvBlock(filters, expand_factor=4, channel_attention=channel_attention, name="stage_3_conv_0")(x)
    # x = fl.FusedMBConvBlock(filters, expand_factor=4, channel_attention=channel_attention, name="stage_3_conv_1")(x)
    # x = fl.FusedMBConvBlock(filters, expand_factor=4, channel_attention=channel_attention, name="stage_3_conv_2")(x)
    x = fl.FusedMBConvBlock(filters, expand_factor=4, strides=2, channel_attention=channel_attention, name="stage_3_conv_3")(x)


    # stage 4 (batch, 32, 32, 64) -> (batch, 16, 16, 128)
    filters = 16 * N
    # x = fl.InverseResidualBlock(filters, strides=1, channel_attention=channel_attention, name="stage_4_conv_0")(x)
    # x = fl.InverseResidualBlock(filters, strides=1, channel_attention=channel_attention, name="stage_4_conv_1")(x)
    # x = fl.InverseResidualBlock(filters, strides=1, channel_attention=channel_attention, name="stage_4_conv_2")(x)
    # x = fl.InverseResidualBlock(filters, strides=1, channel_attention=channel_attention, name="stage_4_conv_3")(x)
    # x = fl.InverseResidualBlock(filters, strides=1, channel_attention=channel_attention, name="stage_4_conv_4")(x)
    x = fl.InverseResidualBlock(filters, strides=2, channel_attention=channel_attention, name="stage_4_conv_5")(x)


    # stage 6 (batch, 16, 16, 128) -> (batch, 8, 8, 256)
    filters = 32 * N
    # x = fl.InverseResidualBlock(filters, strides=1, channel_attention=channel_attention, name="stage_6_conv_0")(x)
    x = fl.InverseResidualBlock(filters, strides=2, channel_attention=channel_attention, name="stage_6_conv_1")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

def classification_head(num_classes=1000, input_shape=(None, 32, 32, 64)):
        n = 480
        inputs = layers.Input(shape=input_shape)
        x = inputs
        x = layers.Conv2D(n, (1, 1), strides=(1, 1))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("hard_silu")(x)

        # Pooling layer
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Reshape((1, 1, n))(x)

        x = layers.Conv2D(1280, (1, 1), strides=(1, 1))(x)
        x = layers.Activation("hard_silu")(x)

        # Final layer
        x = layers.Conv2D(num_classes, (1, 1), strides=(1, 1))(x)
        x = layers.Flatten(name="class_out")(x)

        return tf.keras.Model(inputs=inputs, outputs=x)
