import tensorflow as tf 

class SqueezeExcite(layers.Layer): 
    """Applies squeeze and excitation to input feature maps as seen in https://arxiv.org/abs/1709.01507.
    Args: ratio: The ratio with which the feature map needs to be reduced in the reduction phase. 
    Inputs: Convolutional features. 
    Outputs: Attention modified feature maps. """
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
        
        self.reduction = layers.Dense( units=filters // self.ratio, activation="relu", use_bias=False, )
        self.excite = layers.Dense(units=filters, activation="sigmoid", use_bias=False)
        self.multiply = layers.Multiply()
        
    def call(self, x):
        shortcut = x
        x = self.squeeze(x)
        x = self.reduction(x)
        x = self.excite(x)
        x = self.multiply([shortcut, x])
        return x 
