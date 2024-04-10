import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

class DIV2KDataset(Sequence):
    def __init__(self, hr_dir, lr_dir, hr_size, upscale_factor, batch_size=16, shuffle=True):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_size = hr_size  # This is the target size of the HR images
        self.upscale_factor = upscale_factor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.hr_images = self._load_images(hr_dir)
        self.lr_images = self._load_images(lr_dir)
        if self.shuffle:
            self.on_epoch_end()

    def _load_images(self, images_dir):
        images = [os.path.join(images_dir, img) for img in sorted(os.listdir(images_dir)) if self._is_image_file(img)]
        return images

    @staticmethod
    def _is_image_file(filename):
        return any(filename.lower().endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp'])

    def __len__(self):
        return int(np.ceil(len(self.hr_images) / self.batch_size))

    def __getitem__(self, idx):
        hr_batch_paths = self.hr_images[idx * self.batch_size:(idx + 1) * self.batch_size]
        lr_batch_paths = self.lr_images[idx * self.batch_size:(idx + 1) * self.batch_size]

        hr_batch = np.array([self._process_image(img_path, self.hr_size) for img_path in hr_batch_paths])
        lr_batch = np.array([self._process_image(img_path, (self.hr_size[0] // self.upscale_factor, self.hr_size[1] // self.upscale_factor)) for img_path in lr_batch_paths])
        # debugging
        # print(f"HR batch files: {[os.path.basename(path) for path in hr_batch_paths]}")
        # print(f"LR batch files: {[os.path.basename(path) for path in lr_batch_paths]}")
        return lr_batch, hr_batch

    def _process_image(self, img_path, target_size):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        img = cv2.resize(img, (target_size[0], target_size[1]), interpolation=cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        return img

    def on_epoch_end(self):
        if self.shuffle:
            # Create a combined list of pairs
            paired_images = list(zip(self.hr_images, self.lr_images))
            np.random.shuffle(paired_images)
            # Unzip the shuffled pairs back into separate lists
            self.hr_images, self.lr_images = zip(*paired_images)
