import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Sequence
import numpy as np
from tensorflow.keras import backend as K


def pixel_wise_mse_loss(y_true, y_pred):
    # Ensure both tensors are of the same dtype, either float32 or float16
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    return K.mean(K.square(y_pred - y_true), axis=-1)

def supervised_pixel_wise_adversarial_loss(hr_imgs, sr_imgs, discriminator):
    # Generate discriminator scores for real and fake images
    real_scores = discriminator(hr_imgs, training=True)
    fake_scores = discriminator(sr_imgs, training=True)

    # Calculate the pixel-wise adversarial loss based on these scores
    # Assuming the discriminator outputs scores with a shape compatible for loss calculation with HR images
    # This may involve flattening or otherwise processing discriminator outputs to match HR image dimensions if necessary
    loss = K.mean(K.square(real_scores - fake_scores), axis=-1)

    return loss

def gan_loss(y_true, y_pred):
    # y_true: ground truth HR images
    # y_pred: generated SR images by the generator

    # Calculate pixel-wise MSE loss between HR images and generated SR images
    mse_loss = pixel_wise_mse_loss(y_true, y_pred)

    # Calculate the adversarial loss based on the discriminator's evaluation of the generated SR images
    fake_scores = discriminator(y_pred, training=False)  # Evaluate SR images
    # Here you might need to adapt based on how you calculate the adversarial loss. For simplicity:
    adversarial_loss = K.mean(K.square(fake_scores - 1))  # Encourage discriminator to mistake SR images for real

    # Combine the two losses
    combined_loss = mse_loss + adversarial_loss

    return combined_loss


def construct_gan(generator, discriminator):
    # Input to the generator
    lr_input = Input(shape=(128, 128, 3))

    # Generate SR image
    sr_image = generator(lr_input)

    # Use the discriminator to output a pixel-wise matrix
    validity = discriminator(sr_image)

    # Define and compile the GAN model
    gan_model = Model(lr_input, validity)

    return gan_model