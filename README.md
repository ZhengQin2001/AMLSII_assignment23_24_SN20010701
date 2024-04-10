# AMLSII_assignment23_24_SN20010701



This project explores the use of Generative Adversarial Networks (GANs) for the task of Single Image Super-Resolution (SISR), as part of the NTIRE 2017 challenge. Two GAN-based models, SRGAN and ESRGAN, have been implemented and evaluated on the DIV2K dataset to generate super-resolved images that maintain natural details and textures.

## Project Organization

- `DIV2Kdatasets/`: This directory should contain the DIV2K datasets used for training and evaluating the models. (Now empty)
- `model_results/`: Contains the outputs from the models, including super-resolved images and model weights.(Now empty. The model files can be downloaded with this link: https://drive.google.com/drive/folders/1sBPRhJq-AiUnpcv9Z-eb6QrZsmM83zlU?usp=sharing)

## File Descriptions

- `README.md`: Overview of the project, including its organization, file roles, and setup instructions.
- `dataset.py`: Script for loading and preprocessing the DIV2K dataset.
- `loss.py`: Defines the loss functions used by the GAN models.
- `main.py`: The main script to train and evaluate the GAN models.
- `metric.py`: Contains the metrics used to evaluate the super-resolved images.
- `models.py`: Architectures of the SRGAN and ESRGAN models.
- `requirements.txt`: Lists all dependencies needed to run the code.

## Installation

To set up the environment for running the code, please follow these steps:

```bash
pip install -r requirements.txt
```

## Usage

To start training the models, simply run the main script:
```bash
python main.py
```
