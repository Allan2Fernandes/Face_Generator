import os
import tensorflow as tf

def get_list_image_paths(folder_path, limit):
    image_names = os.listdir(folder_path)
    image_paths = []

    for index, name in enumerate(image_names):
        if index >= limit:
            break
        image_path = os.path.join(folder_path, name)
        image_paths.append(image_path)
        pass
    del image_names
    return image_paths



