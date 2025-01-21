# 🐾 Cat vs Dog Image Classification

This repository implements a binary image classification model to distinguish between images of cats and dogs using a Convolutional Neural Network (CNN). The project leverages TensorFlow and Keras for building and training the model.

---

## ✨ Features

- Binary classification of cats and dogs using CNN.
- Data normalization and augmentation to improve model robustness.
- Lightweight architecture designed for efficiency and scalability.
- Achieves high accuracy on unseen test data.
- Includes training and testing pipelines with TensorFlow/Keras.

---

## 🧠 Model Architecture

The CNN model used in this project comprises the following layers:

1. **Input Layer**:
   - Input shape: `(256, 256, 3)` for RGB images.
   - Images are normalized to the range [0, 1].

2. **Convolutional Layers**:
   - Three convolutional layers with ReLU activation and `3x3` kernels.
   - Filters: 32, 64, and 128, respectively.
   - Batch Normalization after each convolutional layer.

3. **Pooling Layers**:
   - MaxPooling layers with `2x2` pool size to reduce spatial dimensions.

4. **Fully Connected Layers**:
   - Flattened the feature maps into a 1D vector.
   - Two dense layers with 128 and 64 neurons, each with ReLU activation.
   - Dropout layers to prevent overfitting.

5. **Output Layer**:
   - Dense layer with 1 neuron and a sigmoid activation for binary classification.

## 🛠 Implementation

### Data Preparation

The data is organized into `train` and `test` directories with labeled subfolders. Images are resized to `256x256` pixels, and the pixel values are normalized to the range [0, 1].

```python
train_dataset = keras.utils.image_dataset_from_directory(
    directory='/content/train',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256, 256)
)

test_dataset = keras.utils.image_dataset_from_directory(
    directory='/content/test',
    labels="inferred",
    label_mode="int",
    batch_size=32,
    image_size=(256, 256)
)

# Normalization function
def normal(image, label):
    image = tf.cast(image / 255.0, tf.float32)
    return image, label

train_dataset = train_dataset.map(normal)
test_dataset = test_dataset.map(normal)


