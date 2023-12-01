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

def main():
    objects = load_items(OBJECTS_FOLDER, TOTAL_OBJECTS, OBJECT_PREFIX)
    scenes = load_items(SCENES_FOLDER, TOTAL_SCENES, SCENE_PREFIX)
    # Iterate through all scens and objects and check if they are None
    for scene in scenes:
        if scene is None:
            print("Scene is None")
        else:
            print("Scene is not None")
    for obj in objects:
        if obj is None:
            print("Object is None")
        else:
            print("Object is not None")

    cv2.imshow('Scene', scenes[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #obj_detection(objects, scenes)

def load_items(folder, total_items, item_prefix):
    # Load all items in folder into a list
    items = []
    for i in range(1, total_items + 1):
        item_path = os.path.join(folder, f'{item_prefix}{i}.png')
        items.append(cv2.imread(item_path, cv2.IMREAD_COLOR))
    return items


if __name__ == "__main__":
    curr_dir = os.getcwd()
    file_dir = os.path.dirname(os.path.abspath(__file__))

    if curr_dir != file_dir:
        os.chdir(file_dir)
        print(f"Changed the current working directory to: {file_dir}")
    
    main()
