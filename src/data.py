import tensorflow as tf
import random
import glob
import os
import numpy as np
from src.masks import generate_mask

def add_masks(image, img_size=256, mask_type='combined'):
    def _generate(img):
        mask = generate_mask(img_size=img_size, mask_type=mask_type)
        masked_image = img * (1 - mask)
        return masked_image.astype(np.float32), mask.astype(np.float32)
    
    masked_image, mask = tf.py_function(func=_generate, inp=[image], Tout=[tf.float32, tf.float32])
    
    masked_image.set_shape([img_size, img_size, 3])
    mask.set_shape([img_size, img_size, 1])
    
    return (masked_image, mask), image

def load_dataset(image_dir, num_images=40000, img_size=256, batch_size=16, mask_type='combined', seed=42):
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
    dataset = dataset.map(lambda img : add_masks(img, img_size, mask_type), num_parallel_calls=tf.data.AUTOTUNE)
    
    # shuffle, batch, prefetch
    dataset = dataset.shuffle(buffer_size=1000, seed=seed)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
