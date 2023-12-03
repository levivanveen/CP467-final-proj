import json
import cv2
import os
from object_id import scene_objects, update_scene_data, table_data

# Constants
TOTAL_OBJECTS = 18
TOTAL_SCENES = 21
OBJECTS_FOLDER = '../Objects/'
SCENES_FOLDER = '../Scenes/'
DETECTED_FOLDER = '../Detected_Objects/'
KEY_FOLDER = '../Keypoints/'
MATCH_FOLDER = '../Matches/'

OBJECT_PREFIX = 'O'
SCENE_PREFIX = 'S'
IMG_MAP_FILE = 'img_mapping.json'

def main():
    obj_info, scene_info = load_img_mapping()
    objects = load_images(OBJECTS_FOLDER, TOTAL_OBJECTS, OBJECT_PREFIX)
    
    if input("Press enter to get table data") == "":
        table_data(scene_info, TOTAL_OBJECTS)
        return

    # Check the detection for each scene
    i = int(input("Enter scene number: "))
    scene = load_single_img(SCENES_FOLDER, SCENE_PREFIX, i)

    detect_img, scene_key, obj_keys, matches = scene_objects(scene, objects, obj_info)
    update_scene_data(detect_img, scene_info[i - 1])
    # Save scene data
    save_img_mapping(scene_info, obj_info)

    save_img(detect_img, DETECTED_FOLDER, SCENE_PREFIX, f'{i}_detected')
    save_img(scene_key, KEY_FOLDER, SCENE_PREFIX, f'{i}_keypoints')

    for obj_key in obj_keys:
        if obj_key is not None:
            save_img(obj_key['img'], KEY_FOLDER, OBJECT_PREFIX, f'{obj_key["number"]}_keypoints')

    for match in matches:
        if match is not None:
            save_img(match['img'], MATCH_FOLDER, SCENE_PREFIX, f'{i}_O{match["number"]}_matches')

    print("Scene", i, "done")

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

def load_img_mapping():
    with open(IMG_MAP_FILE, 'r') as f:
        # image info is a dict with keys 'objects' and 'scenes'
        image_info = json.load(f)
        return image_info['objects'], image_info['scenes']

def save_img(img, folder, item_prefix, item_id):
    item_path = os.path.join(folder, f'{item_prefix}{item_id}.png')
    cv2.imwrite(item_path, img)

def save_img_mapping(scene_info, obj_info):
    with open(IMG_MAP_FILE, 'w') as f:
        # image info is a dict with keys 'objects' and 'scenes'
        image_info = {
            'objects': obj_info,
            'scenes': scene_info
        }
        json.dump(image_info, f, indent=4)

if __name__ == "__main__":
    curr_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))

    if curr_dir != file_dir:
        os.chdir(file_dir)
        print(f"Changed the current working directory to: {file_dir}")
    
    if main() != 0:
        print("An error occurred while running the program.")
