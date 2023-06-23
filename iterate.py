import networkx as nx
import numpy as np
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from cellpose import models
import imageio as iio
import torch
import matplotlib.pyplot as plt

# Load your TIFF image with two pages
image_path = '/Users/rishi/Desktop/cells.tif'
image = iio.v3.imread(image_path)

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
masks_green, _, _, _ = model.evalq(green_channel, diameter=None, flow_threshold=None)

# Get the centroids of the red and green cells
properties_red = regionprops(masks_red)
centroids_red = np.array([prop.centroid for prop in properties_red])
properties_green = regionprops(masks_green)
centroids_green = np.array([prop.centroid for prop in properties_green])

# Define a function to calculate Euclidean distance
def euclidean_distance(pos1, pos2):
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))

# Iterate over thresholds

# Create an empty graph
G = nx.Graph()

# Add nodes from red and green centroids
for i, centroid in enumerate(centroids_red):
     G.add_node(i, color='red', pos=centroid)

for i, centroid in enumerate(centroids_green, start=len(centroids_red)):  # Starting index after red centroids
    G.add_node(i, color='green', pos=centroid)

# Add edges (with color) between nodes if their distance is below the current threshold

threshold = 34

for i, data_i in G.nodes(data=True):
    for j, data_j in G.nodes(data=True):
        if i != j:  # Avoid self-loops
            dist = euclidean_distance(data_i['pos'], data_j['pos'])
            if dist < threshold:
                G.add_edge(i, j, color='blue' if data_i['color'] != data_j['color'] else data_i['color'])

    # Calculate attribute assortativity for color
    assortativity = nx.attribute_assortativity_coefficient(G, 'color')
    print(f'Assortativity with threshold {threshold}:', assortativity)

from scipy.spatial.distance import pdist, squareform

# Combine red and green centroids
all_centroids = np.concatenate((centroids_red, centroids_green), axis=0)

# Compute pairwise distances
distances = pdist(all_centroids)

# Convert to a square symmetric matrix
dist_matrix = squareform(distances)

# Plot histogram of distances
plt.hist(dist_matrix.ravel(), bins=50)
plt.title('Histogram of pairwise distances')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.show()
