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

# Display the red channel
plt.imshow(red_channel, cmap='bone')
plt.title("Red Channel")
plt.show()

plt.imshow(green_channel, cmap='cool')
plt.title("Green Channel")
plt.show() 

# Define the model
model = models.Cellpose(gpu=True, model_type='cyto', device=torch.device('mps'))

# Apply the model to the red and green channels
masks_red, _, _, _ = model.eval(red_channel, diameter=None, flow_threshold=None)
masks_green, _, _, _ = model.eval(green_channel, diameter=None, flow_threshold=None)
print(masks_red)
print(masks_green)

# Get the centroids of the red and green cells
properties_red = regionprops(masks_red)
centroids_red = np.array([prop.centroid for prop in properties_red])

properties_green = regionprops(masks_green)
centroids_green = np.array([prop.centroid for prop in properties_green])

# Calculate the pairwise distances between the red cells, between the green cells, and between red and green cells
distances_red = cdist(centroids_red, centroids_red)
distances_green = cdist(centroids_green, centroids_green)
distances_red_green = cdist(centroids_red, centroids_green)

# Get the mean distances
mean_distance_red = distances_red.mean()
print(mean_distance_red)
mean_distance_green = distances_green.mean()
print(mean_distance_green)
mean_distance_red_green = distances_red_green.mean()
print(mean_distance_red_green)

# Check if positive assortment is occurring
if mean_distance_red < mean_distance_red_green and mean_distance_green < mean_distance_red_green:
    print("Positive assortment is occurring.")
else:
    print("Positive assortment is not occurring.")

image_combined = np.stack((red_channel, green_channel, np.zeros_like(red_channel)), axis=-1)



# Display image with lines connecting centroids
fig, ax = plt.subplots()
ax.imshow(image_combined, alpha=0.5)
plt.show()


import networkx as nx
import itertools

# Create a new graph
G = nx.Graph()

# Add nodes with color attribute
for i, centroid in enumerate(np.vstack((centroids_red, centroids_green))):
    G.add_node(i, color='red' if i < len(centroids_red) else 'green')

# Add edges based on some criterion (e.g., Euclidean distance threshold)
threshold = 5  # for instance
for i, j in itertools.combinations(range(len(G.nodes)), 2):
    if np.linalg.norm(np.array(G.nodes[i]) - np.array(G.nodes[j])) < threshold:
        G.add_edge(i, j)

# Compute assortativity
assortativity = nx.attribute_assortativity_coefficient(G, 'color')

# Interpret assortativity
if assortativity > 0:
    print("Positive assortment is occurring.")
    output = 1
elif assortativity < 0:
    print("Negative assortment is occurring.")
    output = -1
else:
    print("Random assortment is occurring.")
    output = 0
