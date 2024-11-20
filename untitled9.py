# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:02:41 2024

@author: mi0025
"""

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Load the image
img_path = "C:\\Users\\mi0025\\Desktop\\beautiful-smooth-haired-red-cat-lies-on-the-sofa-royalty-free-image-1678488026.jpg"
color_image = Image.open(img_path)

# Convert to grayscale
gray_image = color_image.convert('L')

# Convert the image to a tensor
img_tensor = torch.from_numpy(np.array(gray_image)).float().unsqueeze(0).unsqueeze(0)

# Define the sharpening filter
sharpening_filter = torch.tensor([[0.0, -0.5, 0.0],
                                  [-0.5, 3.0, -0.5],
                                  [0.0, -0.5, 0.0]])

# Add the batch and channel dimensions
sharpening_filter = sharpening_filter.unsqueeze(0).unsqueeze(0)

# Apply the convolution
filtered_image_tensor = F.conv2d(img_tensor, sharpening_filter, padding=1)

# Convert the tensor back to an image
filtered_image = Image.fromarray(filtered_image_tensor.squeeze().byte().numpy())

# Save the images
original_image_path = 'original_image.jpg'
gray_image_path = 'gray_image.jpg'
filtered_image_path = 'filtered_image.jpg'

color_image.save(original_image_path)
gray_image.save(gray_image_path)
filtered_image.save(filtered_image_path)

# If you wish to display the images as well (optional):
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Original image
ax[0].imshow(color_image)
ax[0].title.set_text('Original Image')
ax[0].axis('off')

# Grayscale image
ax[1].imshow(gray_image, cmap='gray')
ax[1].title.set_text('Grayscale Image')
ax[1].axis('off')

# Filtered image
ax[2].imshow(filtered_image, cmap='gray')
ax[2].title.set_text('Filtered Image')
ax[2].axis('off')

# Show the plot
plt.tight_layout()
plt.show()