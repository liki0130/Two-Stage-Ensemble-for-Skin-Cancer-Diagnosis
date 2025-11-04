!pip install tensorflow keras pandas numpy

import tensorflow as tf
import pandas as pd
import os
import shutil
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras import layers, models

original_dir = '/content/drive/MyDrive/Mini_Project/HAM10000_images'
metadata_path = '/content/drive/MyDrive/Mini_Project/archive/HAM10000_metadata.csv'
output_dir = '/content/drive/MyDrive/output'

df = pd.read_csv(metadata_path)
malignant_labels = ['mel', 'bcc', 'ak', 'vasc', 'df', 'scc']
df['label'] = df['dx'].apply(lambda x: 'malignant' if x in malignant_labels else 'benign')

os.makedirs(os.path.join(output_dir, 'benign'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'malignant'), exist_ok=True)

for _, row in tqdm(df.iterrows(), total=len(df)):
    src = os.path.join(original_dir, f"{row['image_id']}.jpg")
    dst = os.path.join(output_dir, row['label'], f"{row['image_id']}.jpg")
    if os.path.exists(src):
        shutil.copy(src, dst)

train_ds = tf.keras.utils.image_dataset_from_directory(
    output_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    output_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

def build_model(input_shape=(224, 224, 3), num_classes=2):
    base_model = EfficientNetV2B3(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = True

    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model

model = build_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = ModelCheckpoint(
    '/content/drive/MyDrive/skin_model_checkpoint.h5',
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=[checkpoint_cb]
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.show()



















import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

model = tf.keras.models.load_model('/content/drive/MyDrive/models/efficientnetv2_model.h5', compile=False)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/output',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical'
)

class_names = val_ds.class_names

y_true = []
y_pred_probs = []

for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    y_pred_probs.extend(probs[:, 1])
    y_true.extend(np.argmax(labels.numpy(), axis=1))

y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

from sklearn.metrics import classification_report
print("EfficientNetV2 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
