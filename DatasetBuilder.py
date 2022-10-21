import tensorflow as tf


def get_image_tensors(image_paths, image_size):
    image_tensors = []
    for step, image_path in enumerate(image_paths):
        image_tensor = tf.io.read_file(image_path)
        image_tensor = tf.image.decode_png(image_tensor)
        image_tensor = tf.cast(image_tensor, dtype = tf.float32)
        image_tensor = tf.image.resize(image_tensor, size = (image_size[0], image_size[1]))
        image_tensor = (image_tensor-127.5)/127.5
        image_tensors.append(image_tensor)
        if step%500 == 0:
            print("Converted Images = {}".format(step))
        pass
    return image_tensors

def build_label_dataset(image_tensors, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(image_tensors)
    #dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset

def build_label_dataset_v2(directory_path, target_size, batch_size):
    dimension = target_size[0]
    target_dimension = (dimension, dimension)
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=directory_path,
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode='rgb',
        batch_size=None,
        image_size=target_dimension,
        shuffle=False,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation='bilinear',
        follow_links=False,
        crop_to_aspect_ratio=False)
    dataset = dataset.map(map_dataset).batch(batch_size = batch_size, drop_remainder = True).prefetch(buffer_size = 1)
    return dataset

def map_dataset(datapoint):
    datapoint = tf.cast(datapoint, dtype=tf.float32)
    datapoint = (datapoint-127.5)/127.5
    return datapoint