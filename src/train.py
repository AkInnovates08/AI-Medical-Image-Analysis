import tensorflow as tf
from tensorflow.keras import layers
from model import build_model
import os

def train_model():

    train_dir = "data/chest_xray/train"
    val_dir = "data/chest_xray/val"

    print("🚀 Loading dataset...")

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(224, 224),
        batch_size=16,   # 🔥 reduced for your laptop
        label_mode='binary'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(224, 224),
        batch_size=16,
        label_mode='binary'
    )

    print("✅ Dataset loaded successfully")

    # Normalize
    normalization_layer = layers.Rescaling(1./255)

    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # 🔥 CACHE + PREFETCH (VERY IMPORTANT)
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build model
    model = build_model()

    # 🔥 CALLBACKS (IMPORTANT FOR REAL PROJECT)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_model.h5",
            monitor='val_accuracy',
            save_best_only=True
        )
    ]

    print("🚀 Starting training...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks
    )

    # Save final model
    if not os.path.exists("models"):
        os.makedirs("models")

    model.save("models/pneumonia_model.h5")

    print("✅ Model saved successfully!")

if __name__ == "__main__":
    train_model()