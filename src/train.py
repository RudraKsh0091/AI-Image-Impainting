import tensorflow as tf
from tensorflow import keras
import os
from src.data import get_train_val_split, load_dataset
from src.model import build_pconv_unet
from src import config

train_paths, val_paths = get_train_val_split(config.IMAGE_DIR, num_train=config.NUM_TRAIN, num_val=config.NUM_VAL)

train_ds = load_dataset(train_paths, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, mask_type='combined', fixed_seed=None)
val_ds = load_dataset(val_paths, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, mask_type='combined', fixed_seed=123)

model = build_pconv_unet(img_size=config.IMG_SIZE)
model.load_weights("models/best_model_pconv.weights.h5", by_name=True, skip_mismatch=True)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE))

callbacks = [
    keras.callbacks.ModelCheckpoint("best_model_pconv_perceptual.weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss'),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    keras.callbacks.CSVLogger("training_log_pconv_perceptual.csv"),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS, callbacks=callbacks)