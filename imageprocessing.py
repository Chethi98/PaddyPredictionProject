from rembg import remove
from PIL import Image
import cv2
import numpy as np

# Load the image
img = cv2.imread("D:\\Paddy Project\\CleanDataset - Validation\\BacterialLeafBlight\\DSC_0702.jpg")

# Get the image height and width
height, width = img.shape[:2]

# Define the rotation angle
rotation_angle = 0

# Get the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_angle, 1)

# Perform the rotation
rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

# Create the sharpening kernel
kernel = np.array([[0, -1, 0], [-1,5,-1], [0, -1, 0]])

# Sharpen the image
sharpened_image = cv2.filter2D(rotated_image, -1, kernel)

# Convert NumPy array back to PIL image
sharpened_image_pil = Image.fromarray(cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))

# Remove the background
R = remove(sharpened_image_pil)


# Save the image
new_path = 'D:\Paddy Project\ProcessedDataset\BacterialLeafBlight\\Blb83.png'
R.save(new_path)

print(f"Image saved at {new_path}")
