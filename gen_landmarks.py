import os
import sys
import cv2
from matplotlib import pyplot as plt
import sys
import json
import math
import numpy as np
import base64
import tqdm
from PIL import Image, ImageDraw
from argparse import ArgumentParser
import face_alignment
import json
import face_recognition

def draw_landmarks_withcv2(
        image: np.ndarray,
        points: np.ndarray,
        color: tuple = (0, 255, 0),
        thickness: int = 3) -> np.ndarray:
    result = image.copy()
    for p in points:
        result = cv2.circle(result, tuple(p), thickness, color, thickness // 2,
                            -1)
    return result


def gen_landmark(image_path):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
    example_image = cv2.cvtColor(
        cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    face_landmarks = []
    try:
        face_landmarks = fa.get_landmarks_from_image(example_image)
        print('len of land marks is: {}'.format(len(face_landmarks)))

        if len(face_landmarks) > 1: 
            print('land marks have len qual 2 : ',face_landmarks)

        if len(face_landmarks) == 0: 
            return [] 
        else:
            return np.floor(face_landmarks[0]).astype(np.int32)
    except: 
        print('error') 
    return [] 

def visual_distance(path): 
    image = face_recognition.load_image_file(path) 
    landmark = gen_landmark(path)
    threshold = 1.5
    name = path.split('/')[-2] + "->" + path.split('/')[-1]
    
    point_left = landmark[0]
    point_right = landmark[16]
    point_left_eye = landmark[36]
    point_right_eye = landmark[45] 
    
    
    distance1 = math.sqrt( ((point_left[0] - point_left_eye[0])**2)+((point_left[1]-point_left_eye[1])**2) )
    distance2 = math.sqrt( ((point_right[0]- point_right_eye[0])**2)+((point_right[1]-point_right_eye[1])**2) )
    
    print('distance 1:{:.5f} -> distance 2:{:.5f} division: {:.5f} -> name: {} \n'
          .format(distance1,distance2 ,max(distance1,distance2)/ max(min(distance1,distance2),1), name))

# def main(): 
#     # path = '/home/minglee/Documents/aiProjects/dataset/data_wear_mask/false_image_non_mask/bad/40.png'
#     # save_path = ''
# 
#     dataset_path ='/home/minglee/Documents/aiProjects/dataset/data_wear_mask/false_image_non_mask'
#     save_dataset_path = '/home/minglee/Documents/aiProjects/dataset/data_wear_mask'        
#     unmasked_paths=[]
#     for root, dirs, files in os.walk(dataset_path, topdown=False):
#         for dir in tqdm.tqdm(dirs):
#             fs = os.listdir(root + '/' + dir)
#             for name in fs:
#                 new_root = root.replace(dataset_path, save_dataset_path)
#                 new_root = new_root + '/' + dir
#                 if not os.path.exists(new_root):
#                     os.makedirs(new_root)
# 
#                 imgpath = os.path.join(root,dir, name)
#                 save_imgpath = os.path.join(new_root,name)
#                 visual_distance(imgpath)
# 
#     # image = draw_landmarks_withcv2(image, landmark, (0, 255, 0),thickness=2)
# 
#     # plt.imshow(image)
#     # plt.show()
# if __name__ == '__main__':
#     main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
