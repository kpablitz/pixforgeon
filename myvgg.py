import tensorflow as tf
from keras.layers import MaxPooling2D, Conv2D, Flatten, Dense, BatchNormalization, Dropout, Input 
from keras.models import Model

VGG19=[64, 64, 'MP', 128, 128, 'MP', 256, 256, 256, 256, 'MP', 512, 512, 512, 512, 'MP', 512, 512, 512, 512, 'MP'] 

class MyVGG(tf.keras.Model):
    def __init__(self,input_shape=(None, None, 3), num_classes=1000, dropout_rate=0.5, include_top=True, weights=None):
        super(MyVGG, self).__init__()

        self.include_top = include_top
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Input layer with dynamic input size
        self.inputs = Input(shape=input_shape)

        # Build convolutional layers
        self.conv_layers = self._build_conv_layers()

        # Build fully connected layers if include_top is True and
        # Create the model
        if self.include_top:
            self.fcs_layers = self._build_fc_layers(self.conv_layers)
            self.outputs = Dense(self.num_classes, activation='softmax')(self.fcs_layers)
            self.model = Model(inputs=self.inputs, outputs=self.outputs)
        else:
            self.model = Model(inputs=self.inputs, outputs=self.conv_layers)

        # Load weights in case they are given
        if weights is not None:
            self.model.load_weights(weights)

    #Build Conv layers 
    def _build_conv_layers(self):
        x = self.inputs
        block_number = 1
        conv_number = 1
        for layer_filters in VGG19: 
            conv_layer_name = f'block{block_number}_conv{conv_number}'
            pool_layer_name = f'block{block_number}_pool'
            if type(layer_filters) == int: # Conv Layer
                x = Conv2D(filters= layer_filters, kernel_size= (3, 3), activation='relu', padding='same', name=conv_layer_name)(x)
                conv_number += 1
                #x = BatchNormalization()(x)
            else:
                x = MaxPooling2D((2, 2), strides=(2, 2), name=pool_layer_name)(x)                  
                block_number += 1
                conv_number = 1
        return x

    # Build fully connected layers
    def _build_fc_layers(self, x):
        x = Flatten()(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(dropout_rate=self.dropout_rate)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(dropout_rate=self.dropout_rate)(x)
        return x






# Build vgg model using functions
def build_fc_layers(x,dropout_rate=0.5):
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    return x
def build_conv_layers(inputs, network):
    x = inputs
    block_number = 1
    conv_number = 1
    for filters in network: 
        conv_layer_name = f'block{block_number}_conv{conv_number}'
        pool_layer_name = f'block{block_number}_pool'
        if type(filters) == int: # Conv Layer
            x = Conv2D(filters= filters, kernel_size= (3, 3), activation='relu', padding='same', name=conv_layer_name)(x)
            conv_number += 1
            #x = BatchNormalization()(x)
        else:
            x = MaxPooling2D((2, 2), strides=(2, 2), name=pool_layer_name)(x)                  
            block_number += 1
            conv_number = 1
    vgg_conv_block = x
    return vgg_conv_block    

## fix trainable
def my_vgg19(num_classes=1000,dropout_rate=0.5, include_top= True, weights=None):
    inputs = Input(shape=(None,None,3))
    conv_layers = build_conv_layers(inputs, VGG19)
    if include_top:
        fcs_layers = build_fc_layers(conv_layers,dropout_rate=dropout_rate)
        outputs = Dense(num_classes, activation='softmax')(fcs_layers)
        model = Model(inputs=inputs, outputs=outputs)
        model.load_weights(weights)
    else:
        model = Model(inputs=inputs, outputs=conv_layers)
    if weights is not None:
        model.load_weights(weights)
    return model

