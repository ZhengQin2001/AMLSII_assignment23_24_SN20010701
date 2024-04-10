from dataset import DIV2KDataset
import loss
import models
import tensorflow as tf
import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import metric
from tensorflow.keras.mixed_precision import set_global_policy

@tf.function
def train_generator_step(lr_imgs, hr_imgs):
    with tf.GradientTape() as tape:
        sr_imgs = generator(lr_imgs, training=True)
        loss = loss.pixel_wise_mse_loss(hr_imgs, sr_imgs)
    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss

@tf.function
def train_discriminator_step(lr_imgs, hr_imgs):
    with tf.GradientTape() as tape:
        sr_imgs = generator(lr_imgs, training=False)  # Use generator in inference mode here to save memory
        loss = loss.supervised_pixel_wise_adversarial_loss(hr_imgs, sr_imgs, discriminator)
    gradients = tape.gradient(loss, discriminator.trainable_variables)
    discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
    return loss

def display_results(model, dataset, num_images=4):
    for i in range(num_images):
        lr, hr = dataset[i]  # Get the i-th batch of the dataset
        sr = model.predict(np.expand_dims(lr[0], axis=0))  # Predict the super-resolution image

        # Debug: Print the sizes of the images
        print(f"Size of LR image: {lr[0].shape}")  # Shape of the first LR image in the batch
        print(f"Size of SR image: {sr[0].shape}")  # Shape of the predicted SR image
        print(f"Size of HR image: {hr[0].shape}")  # Shape of the first HR image in the batch
        psnr = metric.NTIRE_PeakSNR_imgs(hr[0], sr[0], upscale_factor)
        ssim = metric.NTIRE_SSIM_imgs(hr[0], sr[0], upscale_factor)

        print(f"PSNR: {psnr}")
        print(f"SSIM: {ssim}")

        # Display the images
        plt.figure(figsize=(15, 5))

        # Original Low-Resolution Image
        plt.subplot(1, 3, 1)
        plt.title("Original LR Image")
        plt.imshow(lr[0])
        plt.axis('off')

        # Super-Resolution Image
        plt.subplot(1, 3, 2)
        plt.title("Super-Resolution Image")
        plt.imshow(np.squeeze(sr), cmap='gray')
        plt.axis('off')

        # Original High-Resolution Image
        plt.subplot(1, 3, 3)
        plt.title("Original HR Image")
        plt.imshow(hr[0])
        plt.axis('off')

        plt.show()


def main():
    # To reduce RAM usage
    set_global_policy('mixed_float16')

    generator = models.generator_network((128, 128, 3), num_res_blocks=2)
    # generator = load_model('/model_results/generator_srgan.keras', custom_objects={'PixelShuffle': models.PixelShuffle})

    discriminator = model.discriminator_network((512, 512, 3))
    # discriminator = load_model('/model_results/discriminator_srgan_bi.keras')

    generator_optimizer = Adam(learning_rate=0.0003, beta_1=0.5)
    discriminator_optimizer = Adam(learning_rate=0.00006, beta_1=0.5)
    gan_optimizer = Adam(learning_rate=0.0003, beta_1=0.5)

    generator.compile(optimizer=generator_optimizer, loss=loss.pixel_wise_mse_loss)

    discriminator.compile(optimizer=discriminator_optimizer, loss=loss.supervised_pixel_wise_adversarial_loss)

    gan = loss.construct_gan(generator, discriminator)

    hr_train_dir = '/DIV2Kdatasets/DIV2K_train_HR'
    lr_train_dir = '/DIV2Kdatasets/DIV2K_train_LR_bicubic/X4'
    hr_valid_dir = '/DIV2Kdatasets/DIV2K_valid_HR'
    lr_valid_dir = '/DIV2Kdatasets/DIV2K_valid_LR_bicubic/X4'

    batch_size = 8
    hr_size = (512, 512)  # The size of your high-resolution images
    upscale_factor = 4

    train_dataset = DIV2KDataset(hr_dir=hr_train_dir, lr_dir=lr_train_dir, hr_size=hr_size, upscale_factor=upscale_factor, batch_size=batch_size)
    valid_dataset = DIV2KDataset(hr_dir=hr_valid_dir, lr_dir=lr_valid_dir, hr_size=hr_size, upscale_factor=upscale_factor, batch_size=batch_size)

    
    epochs = 20  # Define the total number of epochs
    save_path = '/models_srgan_test'  # Update this path as necessary
        
    for epoch in range(epochs):
        print(f'Starting Epoch {epoch+1}')

        gen_losses = []
        disc_losses = []

        # Train
        for lr_imgs, hr_imgs in train_dataset:
            gen_loss = train_generator_step(lr_imgs, hr_imgs)
            disc_loss = train_discriminator_step(lr_imgs, hr_imgs)
            gen_losses.append(gen_loss)
            disc_losses.append(disc_loss)

        # Calculate the average loss over all batches for logging
        avg_gen_loss = np.mean([gl.numpy() for gl in gen_losses])
        avg_disc_loss = np.mean([dl.numpy() for dl in disc_losses])

        # Validate
        val_losses = []
        for lr_imgs, hr_imgs in valid_dataset:
            sr_imgs = generator.predict(lr_imgs)
            val_loss = np.mean(loss.pixel_wise_mse_loss(hr_imgs, sr_imgs).numpy())
            val_losses.append(val_loss)

        avg_val_loss = np.mean(val_losses)

        # Logging
        print(f'Epoch {epoch+1}, Avg Gen Loss: {avg_gen_loss}, Avg Disc Loss: {avg_disc_loss}, Avg Val Loss: {avg_val_loss}')

        # Save models
        generator.save(f'{save_path}/generator_epoch_{epoch+1}.keras')
        discriminator.save(f'{save_path}/discriminator_epoch_{epoch+1}.keras')

    # Assuming `gan` is the combined model for inference or further usage
    gan.save(f'/models_srgan_test/gan_final.keras')
    gan.summary()

def display():
    lr_valid_dir = '/DIV2Kdatasets/DIV2K_valid_LR_bicubic/X4'
    hr_valid_dir = '/DIV2Kdatasets/DIV2K_valid_HR'

    batch_size = 5
    hr_size = (512, 512)  # The size of your high-resolution images
    upscale_factor = 4  # The factor by which the low-resolution images are upscaled

    valid_dataset = DIV2KDataset(hr_dir=hr_valid_dir, lr_dir=lr_valid_dir, hr_size=hr_size, upscale_factor=upscale_factor, batch_size=batch_size)

    model = load_model('/DIV2Kdatasets/generator_esrgan_unknown.keras', custom_objects={'PixelShuffle': models.PixelShuffle})  # Change this path if necessary
    model.summary()
    # Display some results
    display_results(model, valid_dataset)

if __name__ == "__main__":
    main()
    # Call this if you need visualisation
    # display() 
    