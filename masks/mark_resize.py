
import cv2 

image_path = '/home/minglee/Documents/aiProjects/git_clone/wear_mask/masks/1.png'
image_save = '/home/minglee/Documents/aiProjects/git_clone/wear_mask/masks/resize_1.png'
shape = (225,225)

def main(): 
    example_image = cv2.cvtColor(
        cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # example_image = cv2.resize(example_image,shape) 
    
    cv2.imwrite(image_save,cv2.cvtColor(example_image,cv2.COLOR_RGB2RGBA) )


if __name__ == '__main__':
    main()