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