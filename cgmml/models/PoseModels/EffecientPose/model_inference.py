from tensorflow import lite
from helpers import *
import track
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from rembg import remove


def load_model(path):
    '''loads the tf-lite model
        Args:
            path: input model path TF-Lite
        Returns:
            model: tf-lite model (loaded)
            resolution: Resolution that model could work on, to be changed if required.
    '''
    tf_lite_model = path
    model = lite.Interpreter(model_path = tf_lite_model)
    model.allocate_tensors()
    resolution = 368
    return model, resolution

def infer(batch, model):
    '''Perfroms inference on the batch of input.
        Args:
            batch: tf-lite model expects the inputs to be of format (H, W, C, batch_size)
            model: tf-lite model.
            
        Returns:
            batch_outputs: the outputs of tf-lite model.
    '''
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], batch)
    model.invoke()
    batch_outputs = model.get_tensor(output_details[-1]['index'])
    return batch_outputs

def search(p0,direction,mask):
    """Function to find the boundary point on segment mask.
        any point 'p' in direction 'dir' starting from point p0 at distance d is given by
        p=p0+d*dir. if mask value at p is 0 then we have found boundary point.

    """
    direction=direction/(np.linalg.norm(direction)+1e-8)
    for d in range(200):
        p=p0+d*direction
        if p[0]<=0 or p[0]>=mask.shape[0]: 
            p=p0+(d-2)*direction
            return p.astype(int),d-2
            
        elif mask[int(p[0]),int(p[1])]==0:
            p=p0+(d-3)*direction
            return p.astype(int),d-3
    return p0,0

def get_ground_coordinates(image, coordinates_f):
    '''
    Function to get the ground coordinates and refine the head_top, 
    since head top is not detected at boundary. we are finding here boundary head_point.
    '''
    # apply segmentation to remove background and create mask
    image = remove(image)[:,:,:-1].copy()
    image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    image_gray= cv2.GaussianBlur(image_gray,(3,3),0)
    th1,image_gray = cv2.threshold(image_gray,5,255,cv2.THRESH_BINARY)
    image_gray= cv2.GaussianBlur(image_gray,(3,3),0)

    # get new head coordinate by searching boundary point in direction of 'pelvis' to 'thorax' starting from 'thorax'.
    if {'thorax', 'pelvis'} <= coordinates_f.keys():
        head_dir=coordinates_f['thorax']-coordinates_f['pelvis']
        coordinates_f['head_top'],d=search(coordinates_f['thorax'],head_dir,image_gray)
        image_gray = cv2.circle(image_gray,  coordinates_f['head_top'][::-1], radius=2, color=125, thickness=-1)

    # get left_ground coordinate by searching boundary point in direction of 'left_knee' to 'left_ankle' starting from 'left_ankle'.
    if {'left_ankle', 'left_knee'} <= coordinates_f.keys():
        dir_akl=coordinates_f['left_ankle']-coordinates_f['left_knee']
        p_l,d=search(coordinates_f['left_ankle'],dir_akl,image_gray)
        # print(p_l, d)
        
        if d < np.linalg.norm(dir_akl):
            coordinates_f['left_ground']=p_l
            # print('Yes')
            image_gray = cv2.circle(image_gray, coordinates_f['left_ground'][::-1], radius=2, color=125, thickness=-1)

    # get right_ground coordinate by searching boundary point in direction of 'right_knee' to 'right_ankle' starting from 'left_ankle'.
    if {'right_ankle', 'right_knee'} <= coordinates_f.keys():
        dir_akr=coordinates_f['right_ankle']-coordinates_f['right_knee']
        p_r,d=search(coordinates_f['right_ankle'],dir_akr,image_gray)
        if d < np.linalg.norm(dir_akr):
            coordinates_f['right_ground']=p_r
            # print("Ok")
            image_gray = cv2.circle(image_gray, coordinates_f['right_ground'][::-1], radius=2, color=125, thickness=-1)

  
    return coordinates_f

def get_pose_vals(file_path, model, resolution, visualize=True):
    '''Helper function to get the inference, and pose coordinates.
        Args:
            file_path: file path of RGB image.
            model: tf-lite loaded model.
            resolution: Resolution of the particular tf-lite model
            visualization: Just to check if annotated_image has to be generated.
            
        Returns:
            coordinates_updated: updated_coordinates (including feet and head-top)
            annotated_image: image upon with the pose co-ordinates have been marked.
    '''
    from PIL import Image
    img = np.array(Image.open(file_path))
    img_height, img_width= img.shape[:2]
    batch = np.expand_dims(img, axis=0)
    
    # preprocess
    batch = preprocess(batch, resolution, True)
    
    # inference
    batch_outputs = infer(batch, model)
    
    #extract coordinates
    coordinates = [extract_coordinates(batch_outputs[0, ...], img_height, img_width)]
    
    coordinates_f={}
    for coord in coordinates[0]:
        if coord:
            n,x,y=coord
            coordinates_f[n]=np.array([int(y*img.shape[0]),int(x*img.shape[1])])
    

    coordinates_updated = get_ground_coordinates(img, coordinates_f)
    
    if visualize and coordinates_updated:
        annotated_image = track.annotate_image(file_path,coordinates)
    
    
    return coordinates_updated, annotated_image

# +
# For testing only; use main_fn

def main_fn(rgb_fpath, model_path):
    '''helper file to generate the required updated_coordinates and annotated_image'''
    model, resolution = load_model(model_path)
    updated_coordinates, annotated_image = get_pose_vals(rgb_fpath, model, resolution)
    return updated_coordinates, annotated_image

# +
# def main():
#     parser = argparse.ArgumentParser(description='Pose estimates of an RGB image using a TFLite model.')
    
#     # RGB image file path
#     parser.add_argument('image_path', type=str, help='Path to the input RGB image file.')
    
#     # model file path
#     parser.add_argument('model_path', type=str, help='Path to the model file.')
    
#     args = parser.parse_args()

#     # Access the image_path and model_path as strings
#     image_path = args.image_path
#     model_path = args.model_path

#     # Inference
#     print(f'Image Path: {image_path}')
#     print(f'Model Path: {model_path}')
#     model, resolution = load_model(model_path)
#     updated_coordinates, annotated_image = get_pose_vals(image_path, model, resolution)
#     return updated_coordinates, annotated_image

# if __name__ == '__main__':
#     main()

# -



