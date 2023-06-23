import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from cellpose import models
from matplotlib.animation import FuncAnimation
import imageio

# Load your TIFF image with two pages
image_path = '/Users/rishi/Desktop/cropcells.tif'
image = imageio.imread(image_path)

# Split the channels
green_channel = image[0]  # First page (index 0) contains green cells
red_channel = image[1]  # Second page (index 1) contains red cells

# Normalize the channels
green_channel = green_channel / green_channel.max()
red_channel = red_channel / red_channel.max()

# Combine the channels into a single image
combined_image = np.dstack((red_channel, green_channel, np.zeros_like(red_channel)))


# Define the model
model = models.Cellpose(gpu=True, model_type='cyto', device=torch.device('mps'))

# Apply the model to the red and green channels
masks_red, _, _, _ = model.eval(red_channel, diameter=None, flow_threshold=None)
masks_green, _, _, _ = model.eval(green_channel, diameter=None, flow_threshold=None)

# Get the centroids of the red and green cells
properties_red = regionprops(masks_red)
centroids_red = np.array([prop.centroid for prop in properties_red])

properties_green = regionprops(masks_green)
centroids_green = np.array([prop.centroid for prop in properties_green])

# Create a list to hold all the lines to be drawn, with color and style
lines = []

# Add the red lines (red to red)
for i in range(centroids_red.shape[0]):
    for j in range(i + 1, centroids_red.shape[0]):
        line = zip(centroids_red[i][::-1], centroids_red[j][::-1]), 'red', '-', 1
        lines.append(line)

# Add the green lines (green to green)
for i in range(centroids_green.shape[0]):
    for j in range(i + 1, centroids_green.shape[0]):
        line = zip(centroids_green[i][::-1], centroids_green[j][::-1]), 'green', '-', 1
        lines.append(line)

# Add the blue lines (red to green)
for i in range(centroids_red.shape[0]):
    for j in range(centroids_green.shape[0]):
        line = zip(centroids_red[i][::-1], centroids_green[j][::-1]), 'blue', '-', 1
        lines.append(line)

# Display image with lines connecting centroids
fig, ax = plt.subplots()
ax.imshow(combined_image)  # Display the original image (red channel)

def draw_lines(frame):
    if frame < len(lines):
        coords, color, linestyle, zorder = lines[frame]
        line = ax.plot(*coords, color=color, linestyle=linestyle, zorder=zorder, alpha=1.0)[0]
        lines_plotted.append(line)  # Store the plotted line
        del lines[frame]  # Delete the drawn line from the list

    # Adjust transparency of plotted lines
    for line in lines_plotted:
        alpha = line.get_alpha()
        line.set_alpha(alpha - 0.01)  # Reduce the transparency gradually

    # Remove lines with alpha less than or equal to 0
    lines_plotted[:] = [line for line in lines_plotted if line.get_alpha() > 0]

lines_plotted = []  # List to store the plotted lines

# Display image with lines connecting centroids
fig, ax = plt.subplots()
ax.imshow(combined_image)  # Display the original image (red channel)

def draw_lines(frame):
    if frame < len(lines):
        coords, color, linestyle, zorder = lines[frame]
        line = ax.plot(*coords, color=color, linestyle=linestyle, zorder=zorder, alpha=1.0)[0]
        lines_plotted.append(line)  # Store the plotted line
        del lines[frame]  # Delete the drawn line from the list

    # Adjust transparency of plotted lines
    for line in lines_plotted:
        alpha = line.get_alpha()
        if alpha > 0.01:
            line.set_alpha(alpha - 0.01)  # Reduce the transparency gradually
        else:
            line.set_alpha(0)  # Set alpha to 0 if it falls below threshold

    # Remove lines with alpha equal to 0
    lines_plotted[:] = [line for line in lines_plotted if line.get_alpha() > 0]

lines_plotted = []  # List to store the plotted lines

# Display image with lines connecting centroids
fig, ax = plt.subplots()
ax.imshow(combined_image)  # Display the original image (red channel)

# Create the animation
ani = FuncAnimation(fig, draw_lines, frames=len(lines), interval=10)

ani.save('output.mp4', writer='ffmpeg')
plt.show()


