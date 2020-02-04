#
# Child Growth Monitor - Free Software for Zero Hunger
# Copyright (c) 2019 Dr. Christian Pfitzner <christian.pfitzner@th-nuernberg.de> for Welthungerhilfe
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


import numpy as np
import cv2
import pandas as pd

from pyntcloud import PyntCloud
from pyntcloud.io import write_ply

import sys
import os 

import pickle
import logging



from enum import Enum

def fuse_point_cloud(points, rgb_vals, confidence, seg_vals): 
    df = pd.DataFrame(columns=['x', 'y', 'z','red', 'green', 'blue', 'c', 'seg'])

    df['x']     = points[:, 0]                              # saving carthesian coordinates
    df['y']     = points[:, 1]
    df['z']     = points[:, 2]

    df['red']   = rgb_vals[:, 2].astype(np.uint8)           # saving the color
    df['green'] = rgb_vals[:, 1].astype(np.uint8)
    df['blue']  = rgb_vals[:, 0].astype(np.uint8)

    df['c']     = confidence[:].astype(np.float)            # saving the confidence

    df['seg']   = seg_vals[:].astype(np.float)              # saving the segmentation

    new_pc      = PyntCloud(df)
    return new_pc


def write_color_ply(fname, points, color_vals, confidence):
    new_pc = fuse_point_cloud(points, color_vals, confidence)
    write_ply(fname, new_pc.points, as_text=True)





#from cgm_fusion.calibration import get_intrinsic_matrix, get_extrinsic_matrix, get_k, get_intrinsic_matrix_depth

from  cgm_fusion.calibration import *


def apply_projection(points):
    intrinsic  = get_intrinsic_matrix_depth()

    ext_d      = get_extrinsic_matrix_depth(4)

    r_vec      =  ext_d[:3, :3]
    t_vec      = -ext_d[:3, 3]

    k1, k2, k3 = get_k_depth()
    im_coords, _ = cv2.projectPoints(points, r_vec, t_vec, intrinsic[:3, :3], np.array([k1, k2, 0, 0]))

    return im_coords



class Channel(Enum):
    x = 0
    y = 1
    z = 2
    confidence = 3
    red = 4
    green = 5
    blue = 6
    segmentation = 7


def get_depth_channel(ply_path, output_path_np, output_path_png):
    channel = Channel.z
    calibration_file =  '/whhdata/calibration.xml'
    if not os.path.exists(calibration_file):                   # check if the califile exists
        logging.error ('Calibration does not exist')
        return 

    # get a default black image
    height         = 224                                       # todo remove magic numbers               
    width          = 172                                       # todo remove magic numbers
    nr_of_channels = 1
    viz_image = np.zeros((height,width,nr_of_channels), np.float64)

    # get the points from the pointcloud
    # print(ply_path)

    try:
        cloud  = PyntCloud.from_file(ply_path)                 # load the data from the files
    except ValueError as e: 
        logging.error(" Error reading point cloud ")
        logging.error(str(e))
        logging.error(ply_path)
        
        
        
    points = cloud.points.values[:, :3]                        # get x y z
    z      = cloud.points.values[:, 2]                   # get only z coordinate
    z      = (z - min(z)) / (max(z) - min(z))                  # normalize the data to 0 to 1

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration 
    im_coords = apply_projection(points)

    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function
    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < width and y >= 0 and y < height:
            viz_image[x,y] = z[i] #255 #255-255*z[i]

    # resize and  return the image after pricessing
    dim = (180, 240)
    viz_image = cv2.resize(viz_image, dim, interpolation = cv2.INTER_AREA)


    np.save(output_path_np, viz_image)
    # viz_2 = np.load("/tmp/out.npy")

    img_n = cv2.normalize(src=viz_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(output_path_png, img_n) 


    return viz_image


'''
Function to get the depth from a point cloud as an image for visualization
'''
def get_viz_channel(ply_path, channel=Channel.z):

    calibration_file =  '/whhdata/calibration.xml'
    if not os.path.exists(calibration_file):                # check if the califile exists
        logging.error ('Calibration does not exist')
        return 

    # get a default black image
    height         = 224
    width          = 172
    nr_of_channels = 1
    viz_image = np.zeros((height,width,nr_of_channels), np.uint8)

    # get the points from the pointcloud
    try:
        cloud  = PyntCloud.from_file(ply_path)              # load the data from the files
    except ValueError as e: 
        logging.error(" Error reading point cloud ")
        logging.error(str(e))

        
        
    points = cloud.points.values[:, :3]                        # get x y z
    z      = cloud.points.values[:, channel]                   # get only z coordinate
    z      = (z - min(z)) / (max(z) - min(z))                  # normalize the data to 0 to 1

    # iterat of the points and calculat the x y coordinates in the image
    # get the data for calibration 
    im_coords = apply_projection(points)

    # manipulate the pixels color value depending on the z coordinate
    # TODO make this a function
    for i, t in enumerate(im_coords):
        x, y = t.squeeze()
        x = int(np.round(x))
        y = int(np.round(y))
        if x >= 0 and x < width and y >= 0 and y < height:
            viz_image[x,y] = 255*z[i]

    # resize and  return the image after pricessing
    imgScale  = 0.25
    newX,newY = viz_image.shape[1]*imgScale, viz_image.shape[0]*imgScale
    cv2.imwrite('/tmp/depth_visualization.png', viz_image) 

    return viz_image


'''
Function to get the rgb from a point cloud as an image for visualization
'''
def get_viz_rgb(ply_path):
    get_viz_channel(ply_path, channel=Channel.red)



'''
Function to get the confidence from a point cloud as an image for visualization
'''
def get_viz_confidence(ply_path):
    get_viz_channel(ply_path, channel=Channel.confidence)



'''
Function to get the segmentation from a point cloud as an image for visualization
'''
def get_viz_segmentation(ply_path):
    get_viz_channel(ply_path, channel=Channel.segmentation)


