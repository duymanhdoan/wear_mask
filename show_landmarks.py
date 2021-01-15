from PIL import Image, ImageDraw
import face_recognition
import os 
import tqdm 
import face_alignment
from skimage import io
# Load the jpg file into a numpy array

def draw_landmarks(image_path): 
    
    image = face_recognition.load_image_file(image_path)
    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)
    
    for face_landmarks in face_landmarks_list:
        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
    
        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            # d.point(face_landmarks[facial_feature])
            d.line(face_landmarks[facial_feature], width=5)
    
    # Show the picture
    pil_image.show()



def main(): 
    dataset_path ='/home/minglee/Documents/aiProjects/dataset/ouput_dir/image_false_example'
    
    list_folder = []
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for dir in dirs:
            fs = os.listdir(root + '/' + dir)
            for name in fs:
                imgpath = os.path.join(root,dir, name)
                list_folder.append(imgpath)
    path = '/home/minglee/Documents/aiProjects/dataset/ouput_dir/image_false_example/dir/20.png'    
    img_path = draw_landmarks(path)
    
    
if __name__ == '__main__':
    main()