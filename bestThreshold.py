import networkx as nx
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from cellpose import models
from matplotlib.animation import FuncAnimation
import imageio as iio
import os

# Define a function to calculate Euclidean distance
def euclidean_distance(pos1, pos2):
    return np.sqrt(np.sum((np.array(pos1) - np.array(pos2))**2))

from scipy.spatial.distance import pdist, squareform

# Define a function to apply model and calculate centroids
def apply_model_and_get_centroids(image_path):
    # Load your TIFF image with two pages
    image = iio.v3.imread(image_path)

    # Split the channels
    green_channel = image[0]  # First page (index 0) contains green cells
    red_channel = image[1]  # Second page (index 1) contains red cells

    # Normalize the channels
    green_channel = green_channel / green_channel.max()
    red_channel = red_channel / red_channel.max()

    # Define the model
    model = models.Cellpose(gpu=True, model_type='cyto', device=torch.device('mps'))

    # Apply the model to the red and green channels
    print("Applying model to", image_path)
    masks_red, _, _, _ = model.eval(red_channel, diameter=None, flow_threshold=None)
    masks_green, _, _, _ = model.eval(green_channel, diameter=None, flow_threshold=None)

    print("Finished analyzing both channels.")

    # Get the centroids of the red and green cells
    properties_red = regionprops(masks_red)
    centroids_red = np.array([prop.centroid for prop in properties_red])

    properties_green = regionprops(masks_green)
    centroids_green = np.array([prop.centroid for prop in properties_green])

    # Combine red and green centroids
    all_centroids = np.concatenate((centroids_red, centroids_green), axis=0)
    print(len(all_centroids), "cells found.")

    # Compute pairwise distances
    distances = squareform(pdist(all_centroids))

    return centroids_red, centroids_green, all_centroids, distances

# Define a function to calculate assortativity for a given threshold
def calculate_assortativity(centroids_red, centroids_green, all_centroids, distances, threshold):
    # Create a fresh graph for each threshold step
    G = nx.Graph()

    # Add nodes from red and green centroids
    for i, centroid in enumerate(centroids_red):
        G.add_node(i, color='red', pos=centroid)

    for i, centroid in enumerate(centroids_green, start=len(centroids_red)):  # Starting index after red centroids
        G.add_node(i, color='green', pos=centroid)

    # Add edges (with color) between nodes if their distance is below the current threshold
    for i in range(len(all_centroids)):
        for j in range(i + 1, len(all_centroids)):  # Avoid self-loops and duplicate edges
            if distances[i, j] < threshold:
                G.add_edge(i, j,
                           color='blue' if G.nodes[i]['color'] != G.nodes[j]['color'] else G.nodes[i]['color'])

    # Calculate the assortativity coefficient and return it
    assortativity = nx.attribute_assortativity_coefficient(G, 'color')
    return assortativity


# Directory containing the TIFF files
directory = '/Users/rishi/Desktop/TIF'

# Initialize variables
best_threshold = None
best_assortativity = float('-inf')

# Iterate through TIFF files in the directory
try:
    for filename in os.listdir(directory):
        if filename.endswith('.tif'):
            file_path = os.path.join(directory, filename)

        # Apply the model and get the centroids and distances
            centroids_red, centroids_green, all_centroids, distances = apply_model_and_get_centroids(file_path)

        # Initialize an array to store assortativity values
            assortativity_values = []

        # Iterate over thresholds
            for threshold in range(36, 41, 1):
                assortativity = calculate_assortativity(centroids_red, centroids_green, all_centroids, distances, threshold)
                assortativity_values.append(assortativity)
                print("The assortativity for", filename, "at", threshold, "is", assortativity)
except:
    pass


