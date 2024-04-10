import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, Add, BatchNormalization, PReLU, Lambda
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, LeakyReLU, Flatten, Dense, MaxPooling2D
from tensorflow.keras.optimizers import Adam

def CNN_model():
    inputs = Input(shape=(None, None, 3))
    x = Conv2D(64, (9, 9), padding='same')(inputs)
    x = Activation('relu')(x)

    for _ in range(4):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)

    # Upsampling
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(3, (5, 5), padding='same')(x)
    outputs = Activation('relu')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

    model.summary()

    model.save('super_res_model.keras')

    return model

def CNN_model2():
    inputs = Input(shape=(None, None, 3))
    x = Conv2D(64, (9, 9), padding='same')(inputs)
    x = Activation('relu')(x)

    for _ in range(4):
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = Activation('relu')(x)

    # Upsampling
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = Activation('relu')(x)

    x = Conv2D(3, (5, 5), padding='same')(x)
    outputs = Activation('relu')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

    model.summary()

    model.save('super_res_model.keras')

    return model


class PixelShuffle(tf.keras.layers.Layer):
    def __init__(self, scale, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        # TensorFlow should be able to infer the output shape from this operation directly
        return tf.nn.depth_to_space(inputs, self.scale)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'scale': self.scale})
        return config


def residual_block(input_tensor, num_filters):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, input_tensor])
    return x

def generator_network(input_shape, num_filters=32, num_res_blocks=1):
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, kernel_size=9, padding='same')(inputs)
    x = PReLU(shared_axes=[1, 2])(x)

    # Save the output of the first convolution to add back after residual blocks
    conv1_out = x

    # Add B residual blocks
    for _ in range(num_res_blocks):
        x = residual_block(x, num_filters)

    # Add more layers after residual blocks
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, conv1_out])  # Element-wise sum

    # Upsampling
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same')(x)  # Increase number of filters for PixelShuffle
    x = PixelShuffle(scale=2)(x)  # Upsampling by factor of 2
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same')(x)  # Increase number of filters for PixelShuffle
    x = PixelShuffle(scale=2)(x)  # Upsampling by factor of 2
    x = PReLU(shared_axes=[1, 2])(x)

    outputs = Conv2D(3, kernel_size=9, padding='same')(x)  # Output convolution, no activation

    model = Model(inputs=inputs, outputs=outputs)

    return model

def discriminator_network(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, kernel_size=3, strides=1, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)

    # Subsequent convolutional layers with increasing depth and stride 2 for downsampling
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # # Continue with convolutional layers, BN and LeakyReLU (omitting for brevity)
    # x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(negative_slope=0.2)(x)

    # Flatten and Dense layers for classification
    x = Flatten()(x)
    # x = Dense(128)(x)
    x = Dense(64)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Dense(1, activation='sigmoid')(x)

    # Create model
    model = Model(inputs=inputs, outputs=x)
    # model.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss='mean_squared_error', metrics=['accuracy'])
    # it will be compiled afterwards

    return model

def RRDB(input_tensor, num_filters):
    """Represents a Residual-in-Residual Dense Block (RRDB)."""
    # Implementation of RRDB is conceptual.
    # It typically consists of several densely connected convolutional layers.
    x = input_tensor
    for _ in range(3):
        x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)  # ESRGAN uses LeakyReLU in some layers
    # Assume 'x_residual' is the output from the densely connected conv layers
    x_residual = x
    x = Add()([x_residual, input_tensor])
    return x

def generator_network_esrgan(input_shape, num_filters=32, num_rrdb_blocks=1):
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters, kernel_size=9, padding='same')(inputs)
    x = LeakyReLU(alpha=0.2)(x)  # Adjusted activation to LeakyReLU as per ESRGAN's practice

    # Save the output of the first convolution to add back after RRDB blocks
    conv1_out = x

    # Add B RRDB blocks
    for _ in range(num_rrdb_blocks):
        x = RRDB(x, num_filters)

    # Add more layers after RRDB blocks
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = Add()([x, conv1_out])  # Element-wise sum

    # Upsampling
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same')(x)
    x = PixelShuffle(scale=2)(x)  # Upsampling by factor of 2
    x = LeakyReLU(alpha=0.2)(x)  # Adjusted activation
    x = Conv2D(num_filters * 4, kernel_size=3, padding='same')(x)
    x = PixelShuffle(scale=2)(x)  # Upsampling by factor of 2
    x = LeakyReLU(alpha=0.2)(x)

    outputs = Conv2D(3, kernel_size=9, padding='same')(x)  # Output convolution, no activation

    model = Model(inputs=inputs, outputs=outputs)
    return model

