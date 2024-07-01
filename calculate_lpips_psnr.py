import cv2
import numpy as np
import torch
import lpips
from PIL import Image
import torchvision.transforms as transforms

def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr

def calculate_lpips(img1_path, img2_path):
    # Load LPIPS model
    loss_fn = lpips.LPIPS(net='alex') 

    # Load images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure images are the same size
        transforms.ToTensor(),
    ])

    img1 = transform(Image.open(img1_path))
    img2 = transform(Image.open(img2_path))

    # Add batch dimension
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    # Calculate LPIPS
    lpips_value = loss_fn(img1, img2)
    return lpips_value.item()

img1 = "/home/ubuntu/Workspace/phat-intern-dev/VinAI/mvsplat/outputs/test/VinAI/view1/view_test_04/color_nf_1000/0005.png"
img2 = "/home/ubuntu/Workspace/phat-intern-dev/VinAI/mvsplat/outputs/test/VinAI/view1/view_test_04/color_nf_100/0005.png"
# Calculate LPIPS
lpips_value = calculate_lpips(img1, img2)
print(f"LPIPS: {lpips_value}")

# Load images
img1 = cv2.imread(img1)
img2 = cv2.imread(img2)

# Ensure images have the same size
img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

# Convert images to grayscale
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Calculate PSNR
psnr_value = calculate_psnr(img1, img2)
print(f"PSNR: {psnr_value}")
