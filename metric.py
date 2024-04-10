import numpy as np
import cv2
from skimage import img_as_float32
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def read_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def preprocess_images(F, G, scale):
    boundarypixels = 6 + scale
    F = F[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels]
    G = G[boundarypixels:-boundarypixels, boundarypixels:-boundarypixels]
    return F, G

def NTIRE_PeakSNR_imgs(F, G, scale):
    # Ensure input images have the same dimensions
    if F.shape != G.shape:
        raise ValueError("Input images must have the same dimensions.")

    F, G = preprocess_images(F, G, scale)
    F = img_as_float32(F)  # Scale to [0, 1] if necessary
    G = img_as_float32(G)  # Scale to [0, 1] if necessary
    mse = np.mean((F - G) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))  # PSNR calculation

def NTIRE_SSIM_imgs(F, G, scale):
    # Ensure input images have the same dimensions
    if F.shape != G.shape:
        raise ValueError("Input images must have the same dimensions.")

    F, G = preprocess_images(F, G, scale)
    F = img_as_float32(F)  # Scale to [0, 1] if necessary
    G = img_as_float32(G)  # Scale to [0, 1] if necessary
    ssim_vals = [compare_ssim(F[:, :, i], G[:, :, i], data_range=1) for i in range(F.shape[2])]
    return np.mean(ssim_vals)

