
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from utils import load_batch
from custom_loss import focal_loss

# ====================================
# CREATE FOLDERS
# ====================================

os.makedirs("results", exist_ok=True)
os.makedirs("saved_model", exist_ok=True)

# ====================================
# LOAD TRAIN DATA
# ====================================

x_train_list = []
y_train_list = []

for i in range(1, 6):

    images, labels = load_batch(
        f"cifar-10-batches-py/data_batch_{i}"
    )

    x_train_list.append(images)
    y_train_list.append(labels)

x_train = np.concatenate(
    x_train_list,
    axis=0
)

y_train = np.concatenate(
    y_train_list,
    axis=0
)

# ====================================
# LOAD TEST DATA
# ====================================

x_test, y_test = load_batch(
    "cifar-10-batches-py/test_batch"
)

# ====================================
# PREPROCESS
# ====================================

x_train = x_train.astype(
    "float32"
) / 255.0

x_test = x_test.astype(
    "float32"
) / 255.0

x_train = tf.image.resize(
    x_train,
    (128,128)
)

x_test = tf.image.resize(
    x_test,
    (128,128)
)

y_train_cat = to_categorical(
    y_train,
    10
)

y_test_cat = to_categorical(
    y_test,
    10
)

print("Training Images:", x_train.shape)
print("Testing Images :", x_test.shape)

# ====================================
# MOBILENET
# ====================================

base_model = MobileNet(
    weights="imagenet",
    include_top=False,
    input_shape=(128,128,3)
)

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.3)(x)

output = Dense(
    10,
    activation="softmax"
)(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ====================================
# COMPILE
# ====================================

model.compile(
    optimizer="adam",
    loss=focal_loss(),
    metrics=["accuracy"]
)

model.summary()

# ====================================
# TRAIN
# ====================================

history = model.fit(
    x_train,
    y_train_cat,
    epochs=15,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

# ====================================
# PREDICTIONS
# ====================================

predictions = model.predict(
    x_test
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# ====================================
# METRICS
# ====================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average='weighted'
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted'
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted'
)

print("\n====================")
print("ACCURACY :", accuracy)
print("PRECISION:", precision)
print("RECALL   :", recall)
print("F1 SCORE :", f1)
print("====================")

# ====================================
# SAVE METRICS
# ====================================

with open(
    "results/metrics.txt",
    "w"
) as f:

    f.write(
        f"Accuracy : {accuracy}\n"
    )

    f.write(
        f"Precision: {precision}\n"
    )

    f.write(
        f"Recall   : {recall}\n"
    )

    f.write(
        f"F1 Score : {f1}\n"
    )

# ====================================
# CONFUSION MATRIX
# ====================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

fig, ax = plt.subplots(
    figsize=(10,10)
)

disp.plot(ax=ax)

plt.savefig(
    "results/confusion_matrix.png"
)

plt.close()

# ====================================
# ACCURACY GRAPH
# ====================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Train Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid()

plt.savefig(
    "results/accuracy.png"
)

plt.close()

# ====================================
# LOSS GRAPH
# ====================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Train Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid()

plt.savefig(
    "results/loss.png"
)

plt.close()

# ====================================
# SAVE MODEL
# ====================================

model.save(
    "saved_model/mobilenet_cifar10.keras"
)

print(
    "\nModel Saved Successfully"
)
