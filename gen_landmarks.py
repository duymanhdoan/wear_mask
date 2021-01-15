import os
import sys
import cv2
from matplotlib import pyplot as plt
import sys
import json
import numpy as np
import base64
import tqdm
# import args 
from PIL import Image, ImageDraw
from argparse import ArgumentParser
# from masked_face_sdk.mask_generation_utils import generate_masks_base
import face_alignment
import json
# import args
# from masked_face_sdk.mask_generation_utils import \
# (
#     extract_target_points_and_characteristic, 
#     extract_polygon,
#     rotate_image_and_points,
#     draw_landmarks,
#     warp_mask,
#     get_traingulation_mesh_points_indexes,
#     end2end_mask_generation
# )



def gen_landmark(image_path):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')
    example_image = cv2.cvtColor(
        cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    face_landmarks = fa.get_landmarks_from_image(example_image)
    face_landmarks = np.floor(face_landmarks[0]).astype(np.int32)

    return  face_landmarks 

def main(): 
    path = '/home/minglee/Documents/aiProjects/dataset/ouput_dir/false_image_non_mask/dir/0.png'
    save_path = ''
    print(gen_landmask(path, save_path))
    
if __name__ == '__main__':
    main()