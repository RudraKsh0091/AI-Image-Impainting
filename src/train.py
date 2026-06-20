import tensorflow as tf
from tensorflow import keras
import os
from src.data import get_train_val_split, load_dataset
from src.model import build_unet
from src import config

class EpochTracker(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        with open("last_epoch.txt", "w+") as file:
            file.write(f"{epoch+1}")

train_paths, val_paths = get_train_val_split(config.IMAGE_DIR, num_train=config.NUM_TRAIN, num_val=config.NUM_VAL)

train_ds = load_dataset(train_paths, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, mask_type='combined', fixed_seed=None)
val_ds = load_dataset(val_paths, img_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE, mask_type='combined', fixed_seed=123)

model = build_unet(img_size=config.IMG_SIZE)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE), loss='mae')

callbacks = [
    keras.callbacks.ModelCheckpoint("best_model.weights.h5", save_best_only=True, save_weights_only=True, monitor='val_loss'),
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor='val_loss'),
    keras.callbacks.CSVLogger("training_log.csv"),
]

initial_epoch = 0
if os.path.exists("last_epoch.txt"):
    with open("last_epoch.txt", "r") as file:
        initial_epoch = int(file.read())
    model.load_weights("best_model.weights.h5")

history = model.fit(train_ds, validation_data=val_ds, epochs=config.EPOCHS, initial_epoch=initial_epoch, callbacks=callbacks + [EpochTracker()])