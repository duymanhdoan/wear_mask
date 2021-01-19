import os
import sys
import argparse
import numpy as np
import cv2
import tqdm
import face_recognition
import math
import dlib
import PIL.Image
from PIL import Image, ImageFile
from matplotlib import pyplot as plt  
from PIL import Image, ImageDraw
import imutils
from gen_landmarks import gen_landmark
import math

__version__ = '0.3.0'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'masks')
# IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
DEFAULT_IMAGE_PATH = os.path.join(IMAGE_DIR, 'default-mask.png')
BLACK_IMAGE_PATH = os.path.join(IMAGE_DIR, 'black-mask.png')
BLUE_IMAGE_PATH = os.path.join(IMAGE_DIR, 'blue-mask.png')
RED_IMAGE_PATH = os.path.join(IMAGE_DIR, 'red-mask.png')
SKIN_IMAGE_PATH = os.path.join(IMAGE_DIR, 'skin-mask.png')
MEDICAL_IMAGE_PATH = os.path.join(IMAGE_DIR,'resize_1.png')
CROPE_SIZE = 128
       

def rect_to_bbox(rect):
    """获得人脸矩形的坐标信息"""
    # print(rect)
    x = rect[3]
    y = rect[0]
    w = rect[1] - x
    h = rect[2] - y
    return (x, y, w, h)

predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
def face_alignment(faces):
    # 预测关键点
    faces_aligned = []
    for face in faces:
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(face_gray, rec)
        #shape = predictor(np.uint8(face), rec)
        # left eye, right eye, nose, left mouth, right mouth
        order = [36, 45, 30, 48, 54]
        for j in order:
            x = shape.part(j).x
            y = shape.part(j).y
        # 计算两眼的中心坐标
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2, (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        # 计算角度
        angle = math.atan2(dy, dx) * 180. / math.pi
        # 计算仿射矩阵
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
        # 进行仿射变换，即旋转
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))
        faces_aligned.append(RotImg)
    return faces_aligned

def cli(pic_path ,save_pic_path):
    parser = argparse.ArgumentParser(description='Wear a face mask in the given picture.')
    # parser.add_argument('pic_path', default='/Users/wuhao/lab/wear-a-mask/spider/new_lfw/Aaron_Tippin/Aaron_Tippin_0001.jpg',help='Picture path.')
    # parser.add_argument('--show', action='store_true', help='Whether show picture with mask or not.')
    parser.add_argument('--model', default='hog', choices=['hog', 'cnn'], help='Which face detection model to use.')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--black', action='store_true', help='Wear black mask')
    group.add_argument('--blue', action='store_true', help='Wear blue mask')
    group.add_argument('--red', action='store_true', help='Wear red mask')
    args = parser.parse_args()

    if not os.path.exists(pic_path):
        print(f'Picture {pic_path} not exists.')
        sys.exit(1)
    mask_path = DEFAULT_IMAGE_PATH
    unmasked_paths = FaceMasker(pic_path, 2.0, mask_path, True, 'cnn',save_pic_path).mask()
    return unmasked_paths

def draw_landmarks(path, face_landmarks): 

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(path)
    # image = imutils.rotate_bound(image,90)

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    point = []
    for face_landmark in face_landmarks:

        # Print the location of each facial feature in this image
        # for facial_feature in face_landmark.keys():
        #     print("The {} in this face has the following points: {}".format(facial_feature, face_landmark[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmark.keys():
            for cc in face_landmark[facial_feature]: 
                point.append(cc)
            
            # d.point(face_landmark[facial_feature]) 
            # d.line(face_landmark[facial_feature], width=5)            
    
    return np.asarray(point)  
    # Show the picture
    # pil_image.show()
def draw_landmarks_with_face(face_image_np, face_landmarks)-> np.ndarray: 

    pil_image = Image.fromarray(face_image_np)
    d = ImageDraw.Draw(pil_image)
    for face_landmark in face_landmarks:
        for facial_feature in face_landmark.keys():
            # d.point(face_landmark[facial_feature]) 
            d.line(face_landmark[facial_feature], width=3)    
    # pil_image.show()
    # plt.show()
    return pil_image

def draw_landmarks_relotate(path): 

    # Load the jpg file into a numpy array

    face_image_np = face_recognition.load_image_file(path)
    plt.imshow(face_image_np)
    plt.show()
    face_image_np = imutils.rotate_bound(face_image_np,15)
    face_locations = face_recognition.face_locations(face_image_np, model='cnn')
    face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)

    pil_image = Image.fromarray(face_image_np)
    d = ImageDraw.Draw(pil_image)
    for face_landmark in face_landmarks:
        # Print the location of each facial feature in this image
        # for facial_feature in face_landmark.keys():
        #     print("The {} in this face has the following points: {}".format(facial_feature, face_landmark[facial_feature]))
        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmark.keys():
            # d.point(face_landmark[facial_feature]) 
            d.line(face_landmark[facial_feature], width=5)    
            print(face_landmark[facial_feature])        
    
 
    # Show the picture
    pil_image.show()
    
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


def count_points(face_landmarks):
    points = []
    for landmark in face_landmarks: 
        for face_feature in landmark.keys(): 
            for point in landmark[face_feature]:
                points.append(point)
        # print('this keys: {} and value: {} and number of points: {} \n'.format(face_feature, landmark[face_feature], len(landmark[face_feature])))
    return points

def visual_landmarks(path_image, face_landmarks):
    """
    visual landmarks of 68 points in face. 
    """
    face_image_np = face_recognition.load_image_file(path_image)
    
    plt.figure(figsize = (8,8))        
    fl = [{'bottom_lip':face_landmarks[0]['bottom_lip'], 
           'bottom_lip':face_landmarks_rec[0]['bottom_lip']
           }]
    fll = draw_landmarks(path_image ,fl)
    
    image = draw_landmarks_withcv2(face_image_np, fll, color=(255, 0, 0), thickness=2)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def visual2bbox(face_image_np_rec,face_locations_rec,face_image_np,face_locations):
    """
    visual bboxx of face detection 
    """
    print('bbox of rec ', face_locations_rec)
    print('bbox of alinment ', face_locations)
    
    plt.figure(figsize = (8,8))   
    x ,y , width, height = rect_to_bbox(face_locations_rec[0])
    x2 ,y2 = x + width, y + height 
    cv2.rectangle(face_image_np_rec, (x,y), (x2,y2), (0,0,255), 1)
    plt.imshow(face_image_np_rec) 
    
    plt.figure(figsize = (8,8))   
    x ,y , width, height = face_locations[0]
    x2 ,y2 = x + width, y + height 
    cv2.rectangle(face_image_np, (x,y), (x2,y2), (0,0,255), 1)
    plt.imshow(face_image_np) 
    plt.show()
    
    plt.figure(figsize = (8,8))        
    fl = [{'top_lip':face_landmarks[0]['top_lip'] ,
           'bottom_lip':face_landmarks_rec[0]['bottom_lip']
    
           }]
    
    fll = draw_landmarks(self.face_path ,fl)
    
    image = draw_landmarks_withcv2(face_image_np, fll, color=(255, 0, 0), thickness=2)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, threshold, mask_path, show=False, model='cnn',save_path = ''):
        self.face_path = face_path
        self.mask_path = mask_path
        self.save_path = save_path
        self.show = show
        self.model = model
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None
        self.threshold = threshold
    
    def distance(self,landmark):
        point_left = landmark[0]
        point_right = landmark[16]
        point_left_eye = landmark[36]
        point_right_eye = landmark[45]         
        distance1 = math.sqrt( ((point_left[0] - point_left_eye[0])**2)+((point_left[1]-point_left_eye[1])**2) )
        distance2 = math.sqrt( ((point_right[0]- point_right_eye[0])**2)+((point_right[1]-point_right_eye[1])**2) )
        return max(distance1,distance2)/ max(min(distance1,distance2),1) <= self.threshold
    
    def mask(self):
        
        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        landmark =  gen_landmark(self.face_path)
        landmarks = []
        for point in landmark: 
            landmarks.append(tuple(point))
        
        if landmarks == []: 
            return null 
        face_landmarks = [{
            'chin':landmarks[0:17],  # true
            'left_eyebrow':landmarks[17:22], # true
            'right_eyebrow':landmarks[22:27],  # true
            'nose_bridge':landmarks[27:31], # true 
            'nose_tip':landmarks[31:36],
            'left_eye':landmarks[36:42], # true
            'right_eye':landmarks[42:48], # true
            'top_lip':landmarks[48:55] + [landmarks[64]] + [landmarks[63]] + [landmarks[62]] + [landmarks[61]] + [landmarks[60]],
            'bottom_lip':landmarks[54:60] + [landmarks[48]] + [landmarks[60]] + [landmarks[67]] + [landmarks[66]] + [landmarks[65]] + [landmarks[64]]
        }]
        
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)
        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue
        
            found_face = True
            self._mask_face(face_landmark)
        
        unmasked_paths = []
        thres = False
        if found_face: 
            thres = self.distance(landmarks)
        
        if found_face and thres:
            # align
            src_faces = []
            src_face_num = 0
            with_mask_face = np.asarray(self._face_img)
            for (i, rect) in enumerate(face_locations):
                src_face_num = src_face_num + 1
                (x, y, w, h) = rect_to_bbox(rect)
                detect_face = with_mask_face[y:y + h, x:x + w]
                src_faces.append(detect_face)
            # 人脸对齐操作并保存
            faces_aligned = face_alignment(src_faces)
            face_num = 0
            for faces in faces_aligned:
                face_num = face_num + 1
                faces = cv2.cvtColor(faces, cv2.COLOR_RGBA2BGR)
                size = (int(CROPE_SIZE), int(CROPE_SIZE))
                faces_after_resize = cv2.resize(faces, size, interpolation=cv2.INTER_AREA)
                cv2.imwrite(self.save_path, faces_after_resize)
        else:
            #在这里记录没有裁的图片
            # print('Found no face.' + self.save_path)
            unmasked_paths.append(self.save_path)
            # cv2.imwrite(self.save_path, cv2.cvtColor(face_image_np, cv2.COLOR_RGBA2BGR))

        
        return unmasked_paths    

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        nose_point = nose_bridge[len(nose_bridge) * 1 // 4]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2]
        chin_bottom_v = np.array(chin_bottom_point)
        chin_left_point = chin[chin_len // 8]
        chin_right_point = chin[chin_len * 7 // 8]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v))

        # left
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        if  mask_left_width > 0 and new_height > 0:
          mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        if  mask_right_width > 0 and new_height> 0:
          mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        path_splits = os.path.splitext(self.face_path)
        new_face_path = path_splits[0] + '-with-mask' + path_splits[1]
        self._face_img.save(new_face_path)
        print(f'Save to {new_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    """
    generate data from image to wear mask face in picture. 
    """
    dataset_path ='/mnt/DATA/duydmFabbi/dataFace/VN-celeb'
    save_dataset_path = '/mnt/DATA/duydmFabbi/dataFace/data_wearmask'  

    unmasked_paths=[]
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for dir in tqdm.tqdm(dirs):
            fs = os.listdir(root + '/' + dir)
            for name in fs:
                new_root = root.replace(dataset_path, save_dataset_path)
                new_root = new_root + '/' + dir
                if not os.path.exists(new_root):
                    os.makedirs(new_root)
                # deal
    
                imgpath = os.path.join(root,dir, name)
                save_imgpath = os.path.join(new_root,name)
                if os.path.exists(save_imgpath):
                    pass   
                else:                                                                                                                                                                       
                    unmasked_paths = cli(imgpath,save_imgpath)
