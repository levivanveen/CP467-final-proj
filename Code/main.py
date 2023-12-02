import json
import cv2
import os
from object_id import obj_detection

# Constants
TOTAL_OBJECTS = 18
TOTAL_SCENES = 21
OBJECTS_FOLDER = '../Objects/'
SCENES_FOLDER = '../Scenes/'
OBJECT_PREFIX = 'O'
SCENE_PREFIX = 'S'
IMG_INFO_FILE = 'img_mapping.json'

def main():
    obj_info, scene_info = load_img_info()
    objects = load_images(OBJECTS_FOLDER, TOTAL_OBJECTS, OBJECT_PREFIX)
    scene = load_single_img
    
    # Load all scene images
    #scenes = load_images(SCENES_FOLDER, TOTAL_SCENES, SCENE_PREFIX)

    obj_detection(objects[0], scenes[0])

    return 0

def load_images(folder, total_items, item_prefix):
    # Load all items in folder into a list
    items = []
    for i in range(1, total_items + 1):
        item_path = os.path.join(folder, f'{item_prefix}{i}.png')
        items.append(cv2.imread(item_path, cv2.IMREAD_COLOR))
    return items

def load_single_img(folder, item_prefix, item_id):
    # Load a single item from folder
    item_path = os.path.join(folder, f'{item_prefix}{item_id}.png')
    return cv2.imread(item_path, cv2.IMREAD_COLOR)

def load_img_info():
    with open(IMG_INFO_FILE, 'r') as f:
        # image info is a dict with keys 'objects' and 'scenes'
        image_info = json.load(f)
        return image_info['objects'], image_info['scenes']


if __name__ == "__main__":
    curr_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))

    if curr_dir != file_dir:
        os.chdir(file_dir)
        print(f"Changed the current working directory to: {file_dir}")
    
    if main() != 0:
        print("An error occurred while running the program.")
