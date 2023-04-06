from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import data, util, io
import numpy as np

img1 = io.imread("image_log/val/reconstruction_gs-00000000.png")
img2 = io.imread("image_log/val/target_view_gs-00000000.png")

# print(img1)
psnr = peak_signal_noise_ratio(img1, img2)
ssim = structural_similarity(img1, img2, channel_axis=-1)

print(psnr)
print(ssim)
