"""
Training script for the baseline CNN model.
Used in AWS SageMaker script-mode training jobs.
"""

import os, argparse
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_baseline(img_size=150, hidden_units=128, num_classes=6):
    return models.Sequential([
        layers.Conv2D(32,(3,3),activation='relu',input_shape=(img_size,img_size,3)),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(hidden_units,activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes,activation='softmax')
    ])

def main(args):
    # SageMaker channels (fallback to env vars)
    train_dir = args.train or os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    val_dir   = args.val   or os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation")
    sm_model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    img_size   = args.img_size
    batch_size = args.batch_size
    epochs     = args.epochs

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.2, horizontal_flip=True
    )
    val_datagen   = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        directory=train_dir, target_size=(img_size,img_size),
        batch_size=batch_size, class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        directory=val_dir, target_size=(img_size,img_size),
        batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    num_classes = len(train_gen.class_indices)

    # Model
    model = build_baseline(img_size=img_size, hidden_units=args.hidden_units, num_classes=num_classes)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    # Callbacks: EarlyStopping + Checkpoint to /opt/ml/model
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(sm_model_dir, "best.keras"),
        save_best_only=True, monitor="val_loss", mode="min"
    )
    rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)

    # Train
    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[es, rlrop, ckpt], verbose=1)

    # Save final export in SavedModel format to SM_MODEL_DIR (SageMaker will tar this)
    tf.saved_model.save(model, sm_model_dir)

    # Also accept --model_dir if container passes it (optional extra save)
    if args.model_dir:
        try:
            tf.saved_model.save(model, args.model_dir)
        except Exception:
            pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default=None)
    p.add_argument("--val", type=str, default=None)
    p.add_argument("--img_size", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden_units", type=int, default=128)
    p.add_argument("--model_dir", type=str, default=None)  # <-- accept model_dir
    args = p.parse_args()
    main(args)