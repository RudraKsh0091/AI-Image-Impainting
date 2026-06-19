import tensorflow as tf
import random
import glob
import os
import numpy as np
from src.masks import generate_mask

def get_train_val_split(image_dir, num_train=36000, num_val=4000, seed=42):
    all_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    random.seed(seed)
    random.shuffle(all_paths)
    val_paths = all_paths[:num_val]
    train_paths = all_paths[num_val:num_val + num_train]
    return train_paths, val_paths

def add_masks(image, img_size=256, mask_type='combined'):
    def _generate(img):
        mask = generate_mask(img_size=img_size, mask_type=mask_type)
        masked_image = img * (1 - mask)
        return masked_image.astype(np.float32), mask.astype(np.float32)
    
    masked_image, mask = tf.py_function(func=_generate, inp=[image], Tout=[tf.float32, tf.float32])
    
    masked_image.set_shape([img_size, img_size, 3])
    mask.set_shape([img_size, img_size, 1])
    
    return (masked_image, mask), image

def parse_image(path, img_size=256):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1.0
    return image

def load_dataset(paths, img_size=256, batch_size=16, mask_type='combined', fixed_seed=None, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    
    dataset = dataset.map(lambda p: parse_image(p, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    
    if fixed_seed is not None:
        np.random.seed(fixed_seed)
        dataset = dataset.map(lambda img: add_masks(img, img_size, mask_type), num_parallel_calls=1)
        dataset = dataset.cache()
    else:
        dataset = dataset.map(lambda img: add_masks(img, img_size, mask_type), num_parallel_calls=tf.data.AUTOTUNE)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=fixed_seed or 42)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
