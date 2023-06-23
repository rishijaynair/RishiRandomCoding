import os
import tifffile
from cellpose import models, io
import networkx as nx
import numpy as np
import torch
from scipy.spatial.distance import cdist

# Specify the directory containing the TIF files
directory = '/Volumes/ThiccSD/TIFF'

# Create a Cellpose model for cytoplasmic segmentation
model_cyto = models.Cellpose(gpu=True, model_type='cyto', device=torch.device('mps'))

# Create a Cellpose model for nuclear segmentation
model_nuclei = models.Cellpose(gpu=True, model_type='cyto', device=torch.device('mps'))

# Loop through each TIF file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.tif'):
        # Load the TIF image
        file_path = os.path.join(directory, filename)
        print(f'Processing file: {file_path}')
        image = tifffile.imread(file_path)

        # Segment the green cells
        print('Segmenting green cells...')
        masks_green, flows_green, styles_green, diams_green = model_cyto.eval(image, diameter=30, channels=[2, 0])

        # Save the green cell masks to a .seg.npy file
        print("Saving...")
        green_mask_path = os.path.join(directory, filename.replace('.tif', '_green.seg.npy'))
        io.masks_flows_to_seg(image, masks_green, flows_green, diams_green, green_mask_path, channels=[2, 0])

        # Segment the red cells
        print('Segmenting red cells...')
        masks_red, flows_red, styles_red, diams_red = model_nuclei.eval(image, diameter=30, channels=[1, 0])

        # Save the red cell masks to a .seg.npy file
        print("Saving...")
        red_mask_path = os.path.join(directory, filename.replace('.tif', '_red.seg.npy'))
        io.masks_flows_to_seg(image, masks_red, flows_red, diams_red, red_mask_path, channels=[1, 0])

