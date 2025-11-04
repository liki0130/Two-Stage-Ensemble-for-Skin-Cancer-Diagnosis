import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

mobilenet_model = tf.keras.models.load_model('/content/drive/MyDrive/models/mobilenet_model.keras', compile=False)
efficientnet_model = tf.keras.models.load_model('/content/drive/MyDrive/models/efficientnetv2_model.h5', compile=False)
inception_model = tf.keras.models.load_model('/content/drive/MyDrive/models/inceptionv3_finetune.keras', compile=False)

val_ds = tf.keras.utils.image_dataset_from_directory(
    '/content/drive/MyDrive/output',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(224, 224),
    batch_size=32,
    label_mode='int'
)

class_names = val_ds.class_names

y_true = []
for _, labels in val_ds:
    y_true.extend(labels.numpy())
y_true = np.array(y_true)

print("Predicting with MobileNetV2...")
preds_mobilenet = mobilenet_model.predict(val_ds, verbose=0).flatten()

print("Predicting with EfficientNetV2...")
preds_efficientnet = efficientnet_model.predict(val_ds, verbose=0)[:, 1]

print("Predicting with InceptionV3...")
preds_inception = inception_model.predict(val_ds, verbose=0).flatten()

avg_preds = (preds_mobilenet + preds_efficientnet + preds_inception) / 3.0
final_preds = (avg_preds > 0.5).astype(int)

print("\nEnsemble Classification Report:")
print(classification_report(y_true, final_preds, target_names=class_names))
