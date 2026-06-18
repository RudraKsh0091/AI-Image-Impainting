import numpy as np
import cv2

def generate_rectangle_mask(img_size=256):
    mask = np.zeros((img_size, img_size, 1), dtype=np.float32)
    top = np.random.randint(0, img_size)
    left = np.random.randint(0, img_size)
    min_size = img_size // 5
    max_size = img_size // 2
    w = np.random.randint(min_size, max_size)
    h = np.random.randint(min_size, max_size)
    bottom = min(top + h, img_size)
    right = min(left + w, img_size)
    cv2.rectangle(mask, (left, top), (right, bottom), 1.0, -1)
    return mask

def generate_brush_mask(img_size=256):
    mask = np.zeros((img_size, img_size, 1), dtype=np.float32)
    num_strokes = np.random.randint(1, 5)
    for _ in range(num_strokes):
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        num_steps = np.random.randint(4, 11)
        brush_thickness = np.random.randint(10, 41)
        for __ in range(num_steps):
            inc_x = np.random.randint(-20, 21)
            inc_y = np.random.randint(-20, 21)
            new_x = min(max(x + inc_x, 0), img_size)
            new_y = min(max(y + inc_y, 0), img_size)
            cv2.line(mask, (x, y), (new_x, new_y), 1.0, brush_thickness)
            x, y = new_x, new_y
    return mask

def generate_mask(img_size=256, mask_type='brush'):
    if mask_type == 'brush':
        return generate_brush_mask(img_size=img_size)
    elif mask_type == 'rectangle':
        return generate_rectangle_mask(img_size=img_size)
    elif mask_type == 'combined':
        choice = np.random.choice(['rectangle', 'brush'])
        return generate_mask(mask_type=choice)
    else:
        return generate_brush_mask(img_size=img_size)