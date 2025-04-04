import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class SqueezeExcite(layers.Layer):
    """
    https://keras.io/examples/vision/patch_convnet/

    Applies squeeze and excitation to input feature maps as seen in
    https://arxiv.org/abs/1709.01507.

    Referenced in https://github.com/google/automl/blob/master/efficientnetv2/effnetv2_model.py

    Args:
        ratio: The ratio with which the feature map needs to be reduced in
        the reduction phase.

    Inputs:
        Convolutional features.

    Outputs:
        Attention modified feature maps.
    """

    def __init__(self, ratio, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduction = layers.Conv2D(
            filters // self.ratio,
        kernel_size=1,
        strides=1,
        kernel_initializer = "he_normal",
        padding='same',
        use_bias=True,
        name='reduction',
            activation="relu",
        )
        self.excite = layers.Conv2D(
            filters,
            kernel_size=1,
        strides=1,
        kernel_initializer = "he_normal",
        padding='same',
        use_bias=True,
        name='excitation',
            activation="sigmoid")
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.reduction(x)
        x = self.excite(x)
        x = self.multiply([shortcut, x])
        return x

@tf.keras.utils.register_keras_serializable()
class EfficientChannelAttention(layers.Layer):
    """
    https://blog.paperspace.com/attention-mechanisms-in-computer-vision-ecanet/
    """
    
    def __init__(self, kernel_size, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config

    def build(self, input_shape):
        channels = input_shape[-1]

        if self.kernel_size < 1:
            log2c = float((tf.math.log(tf.constant(channels))/tf.math.log(tf.constant(2.0))).numpy().squeeze())
            self.kernel_size = max(3, ((log2c / 2) // 2) * 2 + 1)
        
        self.squeeze = layers.GlobalAveragePooling2D(keepdims=False)
        self.conv2 = layers.Conv1D(filters=1, kernel_size=self.kernel_size, kernel_initializer = "glorot_normal", padding='same',use_bias=True)
        self.activation2 = layers.Activation("sigmoid")
        self.multiply = layers.Multiply()

    def call(self, x):
        shortcut = x
        x = self.squeeze(x) # (batch, channels)
        x = tf.expand_dims(x, axis=-1) # (batch, channels, 1)
        x = self.conv2(x) # (batch, channels, 1)
        x = tf.transpose(x, [0, 2, 1]) # (batch, 1, channels)
        x = tf.expand_dims(x, 1) # (batch, 1, 1, channels)
        x = self.activation2(x)
        
        x = self.multiply([shortcut, x])
        return x



@tf.keras.utils.register_keras_serializable()
class InverseResidualBlock(layers.Layer):
    """Implements an Inverse Residual Block like in MobileNetV2 and MobileNetV3

    https://stackoverflow.com/a/61334159

    Args:
        features: Number of features.
        expand_factor: factor by witch to expand number of layers
        strides: Stride used in last convolution.
        batch_norm: flag if Batch Normalisation should be used.

    Inputs:
        Convolutional features.

    Outputs:
        Modified feature maps.
    """

    def __init__(
        self, features=16, expand_factor=4, strides=1, batch_norm=True, channel_attention="", **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features
        self.expand_factor = expand_factor
        self.strides = strides
        self.batch_norm = batch_norm
        self.channel_attention = channel_attention 

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "expand_factor": self.expand_factor,
                "strides": self.strides,
                "batch_norm": self.batch_norm,
                "channel_attention": self.channel_attention,
            }
        )
        return config

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.features * self.expand_factor, (1, 1), strides=1,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        if self.batch_norm:
            self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation("relu6")
        self.dwise = layers.DepthwiseConv2D(3, padding="same", strides=self.strides,
            depthwise_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2 = layers.BatchNormalization()
        self.activation2 = layers.Activation("relu6")
        
        if self.channel_attention == "eca":
            self.squeeze_excite = EfficientChannelAttention(kernel_size=3)
        elif self.channel_attention == "se": 
            self.squeeze_excite = SqueezeExcite(ratio=4)
        else:
            self.squeeze_excite = layers.Lambda(lambda x:x) #layers.Layer()
            
        self.conv2 = layers.Conv2D(self.features, (1, 1), strides=1, padding="same",
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn3 = layers.BatchNormalization()

    def call(self, x):
        shortcut = x
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.activation1(x)
        x = self.dwise(x)
        if self.batch_norm:
            x = self.bn2(x)
        x = self.activation2(x)
        x = self.squeeze_excite(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn3(x)
        if (
            # stride check enforces that we don't add residuals when spatial
            # dimensions are None
            self.strides == 1
            and
            # Depth matches
            x.get_shape().as_list()[3] == shortcut.get_shape().as_list()[3]
        ):
            x = tf.keras.layers.Add()([x, shortcut])

        return x







@tf.keras.utils.register_keras_serializable()
class FusedMBConvBlock(layers.Layer):
    def __init__(
        self, features=16, expand_factor=4, strides=1, batch_norm=True, channel_attention="", **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features
        self.expand_factor = expand_factor
        self.strides = strides
        self.batch_norm = batch_norm
        self.channel_attention = channel_attention 

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "expand_factor": self.expand_factor,
                "strides": self.strides,
                "batch_norm": self.batch_norm,
                "channel_attention": self.channel_attention,
            }
        )
        return config

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(
            self.features * self.expand_factor, (3, 3), strides=self.strides, padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
        )
        if self.batch_norm:
            self.bn1 = layers.BatchNormalization()
        self.activation1 = layers.Activation("relu6")
        
        if self.channel_attention == "eca":
            self.squeeze_excite = EfficientChannelAttention(kernel_size=3)
        elif self.channel_attention == "se": 
            self.squeeze_excite = SqueezeExcite(ratio=4)
        else:
            self.squeeze_excite = layers.Lambda(lambda x:x) #layers.Layer()
            
        self.conv2 = layers.Conv2D(self.features, (1, 1), strides=1, padding="same",
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2 = layers.BatchNormalization()

    def call(self, x):
        shortcut = x
        x = self.conv1(x)
        if self.batch_norm:
            x = self.bn1(x)
        x = self.activation1(x)
        x = self.squeeze_excite(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn2(x)
        if (
            # stride check enforces that we don't add residuals when spatial
            # dimensions are None
            self.strides == 1
            and
            # Depth matches
            x.get_shape().as_list()[3] == shortcut.get_shape().as_list()[3]
        ):
            x = tf.keras.layers.Add()([x, shortcut])

        return x







@tf.keras.utils.register_keras_serializable()
class ASPPBlock(layers.Layer):
    """Implements an Inverse Residual Block like in MobileNetV2 and MobileNetV3

    https://stackoverflow.com/a/61334159

    Args:
        features: Number of features.
        expand_factor: factor by witch to expand number of layers
        strides: Stride used in last convolution.
        batch_norm: flag if Batch Normalisation should be used.

    Inputs:
        Convolutional features.

    Outputs:
        Modified feature maps.
    """

    def __init__(
        self, features=16, expand_factor=4, strides=1, batch_norm=True, channel_attention="", dropout_rate=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features
        self.expand_factor = expand_factor
        self.strides = strides
        self.batch_norm = batch_norm
        self.channel_attention = channel_attention
        self.dropout_rate = dropout_rate
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "expand_factor": self.expand_factor,
                "strides": self.strides,
                "batch_norm": self.batch_norm,
                "channel_attention": self.channel_attention,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def build(self, input_shape):


        self.conv1_a = layers.Conv2D(
            self.features, (1, 1), strides=1,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        if self.batch_norm:
            self.bn1_a = layers.BatchNormalization()
        self.activation1_a = layers.Activation("relu6")
        self.dwise_a = layers.DepthwiseConv2D(1, dilation_rate=(1, 1), padding="same", strides=self.strides,
            depthwise_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_a = layers.BatchNormalization()
        self.dropout_a = layers.Dropout(self.dropout_rate)
        self.activation2_a = layers.Activation("relu6")

        self.conv1_b = layers.Conv2D(
            self.features, (1, 1), strides=1,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        if self.batch_norm:
            self.bn1_b = layers.BatchNormalization()
        self.activation1_b = layers.Activation("relu6")
        self.dwise_b = layers.DepthwiseConv2D(3, dilation_rate=(2, 2), padding="same", strides=self.strides,
            depthwise_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_b = layers.BatchNormalization()
        self.dropout_b = layers.Dropout(self.dropout_rate)
        self.activation2_b = layers.Activation("relu6")
       
        self.conv1_c = layers.Conv2D(
            self.features, (1, 1), strides=1,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        if self.batch_norm:
            self.bn1_c = layers.BatchNormalization()
        self.activation1_c = layers.Activation("relu6")
        self.dwise_c = layers.DepthwiseConv2D(3, dilation_rate=(3, 3), padding="same", strides=self.strides,
            depthwise_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_c = layers.BatchNormalization()
        self.dropout_c = layers.Dropout(self.dropout_rate)
        self.activation2_c = layers.Activation("relu6")
        
        self.conv1_d = layers.Conv2D(
            self.features, (1, 1), strides=1,
            kernel_initializer="he_normal",
            use_bias=False,
        )
        if self.batch_norm:
            self.bn1_d = layers.BatchNormalization()
        self.activation1_d = layers.Activation("relu6")
        self.dwise_d = layers.DepthwiseConv2D(3, dilation_rate=(5, 5), padding="same", strides=self.strides,
            depthwise_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_d = layers.BatchNormalization()
        self.dropout_d = layers.Dropout(self.dropout_rate)
        self.activation2_d = layers.Activation("relu6")

        self.pool_e = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv1_e = layers.Conv2D(filters=self.features, kernel_size=1, padding='same', kernel_initializer='he_normal', use_bias=False)
        if self.batch_norm:
            self.bn2_e = layers.BatchNormalization()
        self.dropout_e = layers.Dropout(self.dropout_rate)
        self.activation2_e = layers.Activation("relu6")
        self.upsample_e = layers.UpSampling2D(size=input_shape[1:3], interpolation="nearest")
        
        
        
        if self.channel_attention == "eca":
            self.squeeze_excite = EfficientChannelAttention(kernel_size=3)
        elif self.channel_attention == "se": 
            self.squeeze_excite = SqueezeExcite(ratio=4)
        else:
            self.squeeze_excite = layers.Lambda(lambda x:x) #layers.Layer()
            
        self.conv2 = layers.Conv2D(self.features, (1, 1), strides=1, padding="same",
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn3 = layers.BatchNormalization()

    def call(self, x):
        shortcut = x

        xa = self.conv1_a(x)
        if self.batch_norm:
            xa = self.bn1_a(xa)
        xa = self.activation1_a(xa)
        xa = self.dwise_a(xa)
        if self.batch_norm:
            xa = self.bn2_a(xa)
        xa = self.dropout_a(xa)
        xa = self.activation2_a(xa)


        xb = self.conv1_b(x)
        if self.batch_norm:
            xb = self.bn1_b(xb)
        xb = self.activation1_b(xb)
        xb = self.dwise_b(xb)
        if self.batch_norm:
            xb = self.bn2_b(xb)
        xb = self.dropout_b(xb)
        xb = self.activation2_b(xb)


        xc = self.conv1_c(x)
        if self.batch_norm:
            xc = self.bn1_c(xc)
        xc = self.activation1_c(xc)
        xc = self.dwise_c(xc)
        if self.batch_norm:
            xc = self.bn2_c(xc)
        xc = self.dropout_c(xc)
        xc = self.activation2_c(xc)


        xd = self.conv1_d(x)
        if self.batch_norm:
            xd = self.bn1_d(xd)
        xd = self.activation1_d(xd)
        xd = self.dwise_d(xd)
        if self.batch_norm:
            xd = self.bn2_d(xd)
        xd = self.dropout_d(xd)
        xd = self.activation2_d(xd)

        xe = self.pool_e(x)
        xe = self.conv1_e(xe)
        if self.batch_norm:
            xe = self.bn2_e(xe)
        xe = self.dropout_e(xe)
        xe = self.activation2_e(xe)
        xe = self.upsample_e(xe)


        x = layers.Concatenate()([xa, xb, xc, xd, xe])
        x = self.squeeze_excite(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn3(x)
        if (
            # stride check enforces that we don't add residuals when spatial
            # dimensions are None
            self.strides == 1
            and
            # Depth matches
            x.get_shape().as_list()[3] == shortcut.get_shape().as_list()[3]
        ):
            x = tf.keras.layers.Add()([x, shortcut])

        return x







@tf.keras.utils.register_keras_serializable()
class FusedASPPBlock(layers.Layer):
    """Implements an Inverse Residual Block like in MobileNetV2 and MobileNetV3

    https://stackoverflow.com/a/61334159

    Args:
        features: Number of features.
        expand_factor: factor by witch to expand number of layers
        strides: Stride used in last convolution.
        batch_norm: flag if Batch Normalisation should be used.

    Inputs:
        Convolutional features.

    Outputs:
        Modified feature maps.
    """

    def __init__(
        self, features=16, expand_factor=4, strides=1, batch_norm=True, channel_attention="", dropout_rate=0.1, **kwargs
    ):
        super().__init__(**kwargs)
        self.features = features
        self.expand_factor = expand_factor
        self.strides = strides
        self.batch_norm = batch_norm
        self.channel_attention = channel_attention
        self.dropout_rate = dropout_rate
        
    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "features": self.features,
                "expand_factor": self.expand_factor,
                "strides": self.strides,
                "batch_norm": self.batch_norm,
                "channel_attention": self.channel_attention,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def build(self, input_shape):


        # self.conv1 = layers.Conv2D(
        #     self.features, (1, 1), strides=1, use_bias=True
        # )
        # if self.batch_norm:
        #     self.bn1 = layers.BatchNormalization()
        # self.activation1 = layers.Activation("relu6")
        
        self.conv_a = layers.Conv2D(self.features, 1, dilation_rate=(1, 1), padding="same", strides=self.strides,
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_a = layers.BatchNormalization()
        self.dropout_a = layers.Dropout(self.dropout_rate)
        self.activation2_a = layers.Activation("relu6")

        # self.conv1_b = layers.Conv2D(
        #     self.features, (1, 1), strides=1, use_bias=True
        # )
        # if self.batch_norm:
        #     self.bn1_b = layers.BatchNormalization()
        # self.activation1_b = layers.Activation("relu6")
        self.conv_b = layers.Conv2D(self.features, 3, dilation_rate=(2, 2), padding="same", strides=self.strides,
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_b = layers.BatchNormalization()
        self.dropout_b = layers.Dropout(self.dropout_rate)
        self.activation2_b = layers.Activation("relu6")
       
        # self.conv1_c = layers.Conv2D(
        #     self.features, (1, 1), strides=1, use_bias=True
        # )
        # if self.batch_norm:
        #     self.bn1_c = layers.BatchNormalization()
        # self.activation1_c = layers.Activation("relu6")
        self.conv_c = layers.Conv2D(self.features, 3, dilation_rate=(3, 3), padding="same", strides=self.strides,
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_c = layers.BatchNormalization()
        self.dropout_c = layers.Dropout(self.dropout_rate)
        self.activation2_c = layers.Activation("relu6")
        
        # self.conv1_d = layers.Conv2D(
        #     self.features, (1, 1), strides=1, use_bias=True
        # )
        # if self.batch_norm:
        #     self.bn1_d = layers.BatchNormalization()
        # self.activation1_d = layers.Activation("relu6")
        self.conv_d = layers.Conv2D(self.features, 3, dilation_rate=(5, 5), padding="same", strides=self.strides,
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn2_d = layers.BatchNormalization()
        self.dropout_d = layers.Dropout(self.dropout_rate)
        self.activation2_d = layers.Activation("relu6")

        self.pool_e = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv1_e = layers.Conv2D(filters=self.features, kernel_size=1, padding='same', kernel_initializer='he_normal', use_bias=False)
        if self.batch_norm:
            self.bn2_e = layers.BatchNormalization()
        self.dropout_e = layers.Dropout(self.dropout_rate)
        self.activation2_e = layers.Activation("relu6")
        self.upsample_e = layers.UpSampling2D(size=input_shape[1:3], interpolation="nearest")
        

        
        
        if self.channel_attention == "eca":
            self.squeeze_excite = EfficientChannelAttention(kernel_size=3)
        elif self.channel_attention == "se": 
            self.squeeze_excite = SqueezeExcite(ratio=4)
        else:
            self.squeeze_excite = layers.Lambda(lambda x:x) #layers.Layer()
            
        self.conv2 = layers.Conv2D(self.features, (1, 1), strides=1, padding="same",
            kernel_initializer="he_normal",
            use_bias=False,)
        if self.batch_norm:
            self.bn3 = layers.BatchNormalization()

    def call(self, x):
        shortcut = x
        
        # x = self.conv1(x)
        # if self.batch_norm:
        #     x = self.bn1(x)
        # x = self.activation1(x)
        
        xa = self.conv_a(x)
        if self.batch_norm:
            xa = self.bn2_a(xa)
        xa = self.dropout_a(xa)
        xa = self.activation2_a(xa)


        # xb = self.conv1_b(x)
        # if self.batch_norm:
        #     xb = self.bn1_b(xb)
        # xb = self.activation1_b(xb)
        xb = self.conv_b(x)
        if self.batch_norm:
            xb = self.bn2_b(xb)
        xb = self.dropout_b(xb)
        xb = self.activation2_b(xb)


        # xc = self.conv1_c(x)
        # if self.batch_norm:
        #     xc = self.bn1_c(xc)
        # xc = self.activation1_c(xc)
        xc = self.conv_c(x)
        if self.batch_norm:
            xc = self.bn2_c(xc)
        xc = self.dropout_c(xc)
        xc = self.activation2_c(xc)


        # xd = self.conv1_d(x)
        # if self.batch_norm:
        #     xd = self.bn1_d(xd)
        # xd = self.activation1_d(xd)
        xd = self.conv_d(x)
        if self.batch_norm:
            xd = self.bn2_d(xd)
        xd = self.dropout_d(xd)
        xd = self.activation2_d(xd)

        xe = self.pool_e(x)
        xe = self.conv1_e(xe)
        if self.batch_norm:
            xe = self.bn2_e(xe)
        xe = self.dropout_e(xe)
        xe = self.activation2_e(xe)
        xe = self.upsample_e(xe)




        x = layers.Concatenate()([xa, xb, xc, xd, xe])
        x = self.squeeze_excite(x)
        x = self.conv2(x)
        if self.batch_norm:
            x = self.bn3(x)
        if (
            # stride check enforces that we don't add residuals when spatial
            # dimensions are None
            self.strides == 1
            and
            # Depth matches
            x.get_shape().as_list()[3] == shortcut.get_shape().as_list()[3]
        ):
            x = tf.keras.layers.Add()([x, shortcut])

        return x













@tf.keras.utils.register_keras_serializable()
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        filters = input_shape[1][-1]
        self.conv_g = layers.Conv2D(filters, 1, padding="same")
        self.bn_g = layers.BatchNormalization()

        self.conv_s = layers.Conv2D(filters, 1, padding="same")
        self.bn_s = layers.BatchNormalization()

        self.add = layers.Add()
        self.act_1 = layers.Activation("relu")
        self.conf_out = layers.Conv2D(1, 1, padding="same")
        self.act_2 = layers.Activation("sigmoid")
        self.mul = layers.Multiply()

    def call(self, x):
        g = x[0]
        s = x[1]

        g = self.conv_g(g)
        g = self.bn_g(g)

        s = self.conv_s(s)
        s = self.bn_s(s)

        out = self.add([g, s])
        out = self.act_1(out)
        out = self.conf_out(out)
        out = self.act_2(out)
        out = self.mul([out, x[1]])
        return out

    def get_config(self):
        config = super().get_config()
        return config


@tf.keras.utils.register_keras_serializable()
class MLP(layers.Layer):
    def __init__(self, hidden_units=[128, 64], dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = []
        for units in hidden_units:
            self.hidden_units.append(
                layers.Dense(units, activation=tf.keras.activations.gelu)
            )
        self.dropout = layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        return config

    def call(self, x):
        for units in self.hidden_units:
            x = units(x)
        x = self.dropout(x)
        return x


@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.experimental.numpy.expand_dims(
            tf.experimental.numpy.arange(start=0, stop=self.num_patches, step=1), axis=0
        )
        projected_patches = self.projection(patch)
        encoded = projected_patches + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


@tf.keras.utils.register_keras_serializable()
class PatchAndEncode(layers.Layer):
    """https://blog.gopenai.com/understanding-vision-transformers-vit-with-tensorflow-and-keras-a-cifar-100-experiment-b820e08473f8"""

    def __init__(self, num_patches=64, patch_size=4, projection_dim=64):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.projection_dim = projection_dim

    def build(self, input_shape):
        # patching the input image into patches
        self.patching = layers.Conv2D(
            self.projection_dim, kernel_size=self.patch_size, strides=self.patch_size
        )
        self.reshape = layers.Reshape(
            (self.num_patches, self.projection_dim), name="pacthes"
        )

        # Learnable positional embeddings
        self.lambda0 = layers.Lambda(
            lambda x: tf.expand_dims(tf.range(self.num_patches), axis=0)
        )
        self.embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=self.projection_dim
        )

        # Add positional embeddings to patches
        self.add = layers.Add()

    def call(self, x):
        # patching the input image into patches
        patching = self.patching(x)
        patches = self.reshape(patching)

        # Learnable positional embeddings
        lmbd = self.lambda0(patches)
        position_embeddings = self.embedding(lmbd)

        # Add positional embeddings to patches
        patches = self.add([patches, position_embeddings])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


class MakeShape(layers.Layer):
    def __init__(self, shape, **kwargs):
        super().__init__(**kwargs)
        self.shape = shape

    def build(self, input_shape):
        s = self.shape
        self.dense = layers.Dense(s[1] * s[2] * s[3] // input_shape[1])
        self.reshape = layers.Reshape(s[1:])

    def call(self, x):
        x = self.dense(x)
        x = self.reshape(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"num_patches": self.num_patches})
        return config


class TransformerBlock(layers.Layer):
    def __init__(self, num_heads=4, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.projection_dim = projection_dim

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads"})

    def build(self, input_shape):
        # Layer normalization 1.
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        # Create a multi-head attention layer.
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.projection_dim, dropout=0.1
        )
        # Skip connection 1.
        self.add1 = layers.Add()
        # Layer normalization 2.
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        # MLP.
        self.mlp = MLP(hidden_units=[self.projection_dim * 2, self.projection_dim])
        # Skip connection 2.
        self.add2 = layers.Add()

    def call(self, x):
        # Layer normalization 1.
        x1 = self.norm1(x)
        # Create a multi-head attention layer.
        attention_output = self.attention(x1, x1)
        # Skip connection 1.
        x2 = self.add1([attention_output, x])
        # Layer normalization 2.
        x3 = self.norm2(x2)
        # MLP.
        x3 = self.mlp(x3)
        # Skip connection 2.
        encoded_patches = self.add2([x3, x2])
        return encoded_patches
