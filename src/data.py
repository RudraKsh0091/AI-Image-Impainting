import tensorflow as tf
import random
import glob
import os

def load_dataset(image_dir, num_images=40000, img_size=256, batch_size=16, seed=42):
    # collect all jpg paths
    all_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    # random sample
    random.seed(seed)
    sampled_paths = random.sample(all_paths, min(num_images, len(all_paths)))
    
    # create dataset from paths
    dataset = tf.data.Dataset.from_tensor_slices(sampled_paths)
    
    # parse each image
    def parse_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [img_size, img_size])
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1.0
        return image
    
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # shuffle, batch, prefetch
    dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset