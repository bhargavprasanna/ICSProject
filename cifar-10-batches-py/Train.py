import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ==========================
# Load Dataset
# ==========================

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training Images:", x_train.shape)
print("Testing Images:", x_test.shape)
print("Training Labels:", y_train.shape)
print("Testing Labels:", y_test.shape)

# ==========================
# Normalize Images
# ==========================

x_train = x_train / 255.0
x_test = x_test / 255.0

# ==========================
# Resize Images for MobileNet
# ==========================

x_train = tf.image.resize(x_train, (96, 96))
x_test = tf.image.resize(x_test, (96, 96))

# ==========================
# One Hot Encoding
# ==========================

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# ==========================
# Build MobileNet Model
# ==========================

base_model = MobileNet(
    weights=None,
    include_top=False,
    input_shape=(96, 96, 3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)

predictions = Dense(
    10,
    activation='softmax'
)(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

# ==========================
# Model Summary
# ==========================

model.summary()

# ==========================
# Compile Model
# ==========================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================
# Train Model
# ==========================

history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# ==========================
# Evaluate Model
# ==========================

loss, accuracy = model.evaluate(
    x_test,
    y_test
)

print("\nTest Accuracy:", accuracy * 100)
print("Test Loss:", loss)

# ==========================
# Save Model
# ==========================

model.save("mobilenet_cifar10.h5")

print("\nModel Saved Successfully!")

# ==========================
# Accuracy Graph
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title('MobileNet Accuracy on CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.show()

# ==========================
# Loss Graph
# ==========================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title('MobileNet Loss on CIFAR-10')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.show()