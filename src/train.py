import tensorflow as tf
from tensorflow import keras
from src.data import get_train_val_split, load_dataset
from src.model import build_unet
from src.config import *

train_paths, val_paths = get_train_val_split(IMAGE_DIR, num_train=NUM_TRAIN, num_val=NUM_VAL)

train_ds = load_dataset(train_paths, img_size=IMG_SIZE, batch_size=BATCH_SIZE, mask_type='combined', fixed_seed=None)
val_ds = load_dataset(val_paths, img_size=IMG_SIZE, batch_size=BATCH_SIZE, mask_type='combined', fixed_seed=123)

model = build_unet(img_size=IMG_SIZE)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mae')

callbacks = [
    keras.callbacks.ModelCheckpoint("best_model.weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss'),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    keras.callbacks.CSVLogger("training_log.csv"),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)