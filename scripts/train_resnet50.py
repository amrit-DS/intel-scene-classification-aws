"""
Training script for the ResNet50 transfer learning model.
Used in AWS SageMaker script-mode training jobs.
"""

import os, argparse, tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

def main(args):
    train_dir = args.train or os.environ.get("SM_CHANNEL_TRAINING","/opt/ml/input/data/training")
    val_dir   = args.val   or os.environ.get("SM_CHANNEL_VALIDATION","/opt/ml/input/data/validation")
    sm_model_dir = os.environ.get("SM_MODEL_DIR","/opt/ml/model")

    img, bs, epochs, lr = args.img_size, args.batch_size, args.epochs, args.lr

    tr = ImageDataGenerator(preprocessing_function=preprocess_input)
    va = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_gen = tr.flow_from_directory(train_dir, target_size=(img,img), batch_size=bs, class_mode='categorical')
    val_gen   = va.flow_from_directory(val_dir,   target_size=(img,img), batch_size=bs, class_mode='categorical', shuffle=False)

    base = ResNet50(weights='imagenet', include_top=False, input_shape=(img,img,3))
    base.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(train_gen.class_indices), activation='softmax')(x)
    model = models.Model(inputs=base.input, outputs=out)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    es  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    rlp = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1)

    model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=[es, rlp], verbose=1)

    # Save both formats
    tf.saved_model.save(model, sm_model_dir)                     # for SageMaker endpoints
    try: model.save(os.path.join(sm_model_dir, "keras"))         # for easy local tf.keras load
    except Exception: pass

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=str, default=None)
    p.add_argument("--val",   type=str, default=None)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--model_dir", type=str, default=None)
    args = p.parse_args(); main(args)