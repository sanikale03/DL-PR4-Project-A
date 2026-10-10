# 📄 Summary Report

# 🧠 Deep Learning PR-4 — CNN from Scratch: CIFAR-10 Image Classifier

---

## 📌 Project Problem

Image classification is a **multi-class classification** problem where the objective is to identify the category of an image from its visual features.

In this project, the **CIFAR-10 dataset** is used to build and evaluate Convolutional Neural Network (CNN) models for image classification.

The project focuses on understanding:

- Convolution
- Padding
- Strides
- Pooling
- Flattening
- CNN architecture
- Backpropagation
- Model training
- Data augmentation
- Transfer learning
- Feature extraction
- Fine-tuning
- Residual connections

The objective is to build a complete deep learning workflow and compare CNN approaches using **accuracy, loss, confusion matrix and class-wise performance**.

---

# 📂 Dataset Description

The project uses the **CIFAR-10 dataset** supplied through TensorFlow/Keras.

### Dataset Details

- Training Images: **50,000**
- Test Images: **10,000**
- Image Size: **32 × 32 pixels**
- Channels: **3 RGB channels**
- Number of Classes: **10**
- Problem: **Multi-Class Image Classification**

### CIFAR-10 Classes

```text
0 → Airplane
1 → Automobile
2 → Bird
3 → Cat
4 → Deer
5 → Dog
6 → Frog
7 → Horse
8 → Ship
9 → Truck
```

---

# ⚙️ A1. Data Loading, EDA & Preprocessing

The CIFAR-10 dataset is loaded using the built-in Keras dataset loader.

```python
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
```

### Dataset Shapes

```text
X_train → 50,000 images
y_train → 50,000 labels
X_test  → 10,000 images
y_test  → 10,000 labels
```

Each image has the shape:

```text
32 × 32 × 3
```

where 32 represents height and width and 3 represents RGB channels.

### Preprocessing

The preprocessing steps include:

- Dataset loading
- Shape verification
- Class-name definition
- Sample image visualization
- Class distribution analysis
- Pixel-value inspection
- Pixel normalization
- Label encoding

Pixel values are normalized from:

```text
0 → 255
```

to:

```text
0 → 1
```

using:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0
```

For categorical training, labels are converted using:

```python
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)
```

---

# 📊 Exploratory Data Analysis

The project performs:

- Dataset shape verification
- Class distribution analysis
- Sample image visualization
- Pixel-value distribution
- RGB image inspection
- Class-wise sample inspection

A sample grid is generated to visually inspect images from the ten CIFAR-10 classes.

Pixel distributions are also inspected before and after normalization.

Normalization improves numerical stability and helps the neural network train efficiently.

---

# 🧠 CNN vs ANN

For CIFAR-10, every image contains:

```text
32 × 32 × 3 = 3072
```

pixel values.

An ANN normally flattens these values before passing them to dense layers, which loses the original spatial relationships.

### ANN

```text
Image
  ↓
Flatten
  ↓
Dense Layers
  ↓
Output
```

### CNN

```text
Image
  ↓
Convolution
  ↓
Activation
  ↓
Pooling
  ↓
Convolution
  ↓
Pooling
  ↓
Flatten
  ↓
Dense
  ↓
Output
```

CNNs preserve spatial information and learn local visual patterns such as edges, textures and shapes.

---

# 🧩 A2. Convolution Operation, Padding & Strides

## 🔹 Convolution Operation

Convolution is the main operation used by CNNs.

A kernel moves across an image and performs element-wise multiplication followed by summation.

For example, a 3 × 3 kernel operates on a 3 × 3 image region.

```text
Image Region       Kernel

a b c              w1 w2 w3
d e f      ×       w4 w5 w6
g h i              w7 w8 w9
```

The resulting value is:

```text
a*w1 + b*w2 + c*w3
+ d*w4 + e*w5 + f*w6
+ g*w7 + h*w8 + i*w9
```

Different kernels learn different visual features.

---

# 🔲 Manual Convolution

A manual convolution experiment is implemented using a small grayscale image and an edge-detection kernel.

Example:

```text
[-1  -1  -1]
[ 0   0   0]
[ 1   1   1]
```

This demonstrates how convolution can detect edge information.

The experiment helps explain what happens internally inside a CNN convolution layer.

---

# 📐 Padding

Padding adds pixels around the image boundary.

### VALID Padding

```text
Input
 ↓
Convolution
 ↓
Smaller Feature Map
```

No additional pixels are added.

### SAME Padding

```text
Input
 ↓
Padding
 ↓
Convolution
 ↓
Preserved Spatial Size
```

SAME padding helps preserve spatial dimensions.

---

# 🚶 Stride

Stride controls how far the kernel moves during convolution.

### Stride = 1

The filter moves one pixel at a time.

```text
→
→
→
```

### Stride = 2

The filter moves two pixels at a time.

```text
→→
→→
```

A larger stride reduces the spatial size of the resulting feature map.

---

# 🧠 Feature Maps

A convolution layer can contain multiple filters.

Example:

```text
Input Image
     ↓
Conv2D
     ↓
16 Filters
     ↓
16 Feature Maps
```

Early convolution layers generally learn simple patterns such as edges and lines.

Deeper layers learn more complex structures such as shapes, textures and object parts.

---

# 🏊 A3. Pooling, Flatten & CNN Architecture

## 🔹 Max Pooling

Max pooling selects the maximum value from a local region.

Example:

```text
1  3
2  4
```

Output:

```text
4
```

Max pooling helps:

- Reduce spatial dimensions
- Reduce computation
- Retain strong features
- Improve robustness to small translations

---

# 🔹 Average Pooling

Average pooling calculates the average value.

Example:

```text
1  3
2  4
```

Output:

```text
(1 + 3 + 2 + 4) / 4 = 2.5
```

### Max Pooling vs Average Pooling

| Method | Operation | Main Advantage |
|---|---|---|
| Max Pooling | Maximum value | Retains strongest activation |
| Average Pooling | Average value | Produces smoother representation |

---

# 🔄 Flatten Layer

After convolution and pooling, feature maps must be converted into a one-dimensional representation before entering dense layers.

```text
Feature Maps
     ↓
Flatten
     ↓
1D Feature Vector
     ↓
Dense Layers
```

---

# 🏛️ LeNet-5 Architecture

LeNet-5 is a classic CNN architecture.

The architecture follows:

```text
Input
  ↓
Convolution
  ↓
Pooling
  ↓
Convolution
  ↓
Pooling
  ↓
Flatten
  ↓
Fully Connected
  ↓
Output
```

A CIFAR-10 adapted version is implemented for comparison.

---

# 🏗️ Custom CNN Architecture

A custom CNN is developed for CIFAR-10 classification.

```text
Input: 32 × 32 × 3
        ↓
Conv2D
        ↓
ReLU
        ↓
MaxPooling
        ↓
Conv2D
        ↓
ReLU
        ↓
MaxPooling
        ↓
Conv2D
        ↓
ReLU
        ↓
Flatten
        ↓
Dense
        ↓
Dropout
        ↓
Dense(10)
        ↓
Softmax
```

The final layer contains **10 neurons**, corresponding to the ten CIFAR-10 classes.

---

# 🔁 A4. Backpropagation in CNN & Training

Backpropagation updates CNN weights according to the error produced by the model.

The process is:

```text
Input Image
     ↓
Forward Pass
     ↓
Prediction
     ↓
Loss Calculation
     ↓
Backpropagation
     ↓
Gradient Calculation
     ↓
Weight Update
```

This process is repeated for batches and epochs.

---

# 📉 Loss Function

For multi-class classification, **Categorical Crossentropy** is used.

The loss measures the difference between the actual class and predicted class probabilities.

Lower loss generally indicates better agreement between predictions and target labels.

---

# ⚡ Optimizer

The CNN is trained using the **Adam optimizer**.

Adam provides adaptive parameter updates and combines ideas related to momentum and adaptive learning rates.

---

# ⏹️ Training Callbacks

The training workflow uses:

### EarlyStopping

Stops training when validation performance stops improving.

### ReduceLROnPlateau

Reduces the learning rate when validation loss stops improving.

These callbacks help improve training stability and reduce unnecessary training.

---

# 📈 Training & Validation Curves

The notebook generates:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

These curves help identify overfitting and underfitting.

### Underfitting

```text
Training Accuracy → Low
Validation Accuracy → Low
```

### Overfitting

```text
Training Accuracy → High
Validation Accuracy → Lower
```

---

# 📊 Model Evaluation

The trained CNN is evaluated on the test dataset.

The evaluation includes:

- Test loss
- Test accuracy
- Confusion matrix
- Class-wise performance
- Misclassified images

### Confusion Matrix

The confusion matrix shows actual classes against predicted classes and helps identify which categories are commonly confused.

---

# ❌ Misclassified Images

The notebook displays incorrectly classified test images.

Each displayed sample contains:

```text
Actual Label
Predicted Label
```

This helps analyse the visual cases where the model makes incorrect predictions.

Visually similar classes can be more difficult to distinguish.

---

# 🔄 A5. Data Augmentation

Data augmentation creates modified versions of existing training images.

The objective is to increase training diversity and reduce overfitting.

The augmentation experiment includes transformations such as:

- Rotation
- Width shift
- Height shift
- Horizontal flip
- Zoom

Example:

```python
ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
```

---

# 🖼️ Augmented Image Visualization

The notebook displays multiple transformed versions of an original image.

```text
Original Image
      ↓
Rotation
      ↓
Shift
      ↓
Flip
      ↓
Zoom
      ↓
Augmented Images
```

These transformations increase variation in the training data.

---

# 🏋️ Augmented CNN Training

The CNN is retrained using augmented images.

The original and augmented models are compared using:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss
- Test performance

The purpose is to determine whether augmentation improves generalization and reduces overfitting.

---

# 🧱 Keras Data Augmentation Layers

Keras preprocessing layers can also be used.

```python
keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1)
])
```

These transformations are applied during training.

---

# 🚀 A6. Transfer Learning

Transfer learning uses a pretrained CNN as a starting point for a new image-classification task.

In this project, **MobileNetV2** is used for transfer learning.

The architecture is:

```text
CIFAR-10 Image
      ↓
Resize
      ↓
MobileNetV2 Base
      ↓
Global Average Pooling
      ↓
Dense Layer
      ↓
Softmax
      ↓
10 Classes
```

---

# ❄️ Feature Extraction

During feature extraction, the pretrained MobileNetV2 base is frozen.

```text
MobileNetV2
     ↓
Frozen Base
     ↓
Feature Extraction
     ↓
New Classification Head
     ↓
CIFAR-10 Classes
```

Only the newly added classification layers are trained.

---

# 🔓 Fine-Tuning

After feature extraction, selected upper layers of MobileNetV2 are unfrozen.

A very small learning rate is used for fine-tuning.

```text
Learning Rate = 0.00001
```

The process is:

```text
Pretrained Model
      ↓
Freeze Base
      ↓
Train Classification Head
      ↓
Unfreeze Selected Layers
      ↓
Small Learning Rate
      ↓
Fine-Tuning
```

Fine-tuning allows pretrained visual features to adapt to CIFAR-10.

---

# 🧠 Feature Extraction vs Fine-Tuning

| Method | Base Model | Training |
|---|---|---|
| Feature Extraction | Frozen | Classification head |
| Fine-Tuning | Partially unfrozen | Head + selected base layers |

Feature extraction is computationally simpler, while fine-tuning provides additional adaptation to the target dataset.

---

# 🧩 Residual Connection

A residual connection allows information to bypass one or more layers.

```text
Input
  │
  ├──────────────┐
  ↓              │
Conv             │
  ↓              │
ReLU             │
  ↓              │
Conv             │
  ↓              │
Add ←────────────┘
  ↓
ReLU
```

Mathematically:

```text
Output = F(x) + x
```

Residual connections help deep networks learn efficiently and support the training of deeper architectures.

---

# 📊 Model Comparison

The main approaches implemented in the project are:

| Model | Main Technique | Accuracy | Loss |
|---|---|---:|---:|
| LeNet-5 | Classic CNN | Generated | Generated |
| Custom CNN | CNN from Scratch | Generated | Generated |
| Augmented CNN | Data Augmentation | Generated | Generated |
| MobileNetV2 | Feature Extraction | Generated | Generated |
| MobileNetV2 | Fine-Tuning | Generated | Generated |

The actual numerical values should be taken from the executed notebook outputs.

---

# 📈 Complete Graph & Output List

### A1 — Data Loading & EDA

- CIFAR-10 dataset shapes
- Class distribution
- Sample image grid
- Pixel-value distribution

### A2 — Convolution

- Manual convolution output
- VALID padding
- SAME padding
- Stride comparison
- Feature-map demonstration

### A3 — Pooling & Architecture

- Max pooling
- Average pooling
- Flatten demonstration
- LeNet-5 architecture
- Custom CNN architecture

### A4 — Training

- Training accuracy curve
- Validation accuracy curve
- Training loss curve
- Validation loss curve
- Test accuracy
- Test loss
- Confusion matrix
- Misclassified images

### A5 — Data Augmentation

- Augmented image grid
- Original vs augmented curves
- Augmented CNN evaluation

### A6 — Transfer Learning

- MobileNetV2 feature extraction
- Fine-tuning
- Model comparison
- Residual connection demonstration

---

# 🏆 Final Evaluation

The final evaluation compares the implemented models using their generated test results.

Important evaluation outputs include:

```text
Test Accuracy
Test Loss
Confusion Matrix
Class-wise Performance
```

Accuracy provides an overall measure of correct predictions, while the confusion matrix provides detailed information about class-level errors.

---

# 🔍 Key Findings

### 1. CNNs are suitable for image classification

Convolution layers preserve spatial relationships and learn useful visual patterns.

### 2. Padding controls spatial dimensions

SAME padding can preserve spatial dimensions, while VALID padding reduces them.

### 3. Stride controls feature-map size

Increasing stride reduces spatial dimensions and computational requirements.

### 4. Pooling reduces dimensionality

Pooling reduces feature-map size while retaining important information.

### 5. Data augmentation increases training diversity

Augmentation creates additional variations of training images and can improve generalization.

### 6. Transfer learning reuses learned visual features

Pretrained models provide useful representations learned from large image datasets.

### 7. Fine-tuning adapts pretrained features

Fine-tuning selected layers allows the pretrained model to specialize for the target dataset.

### 8. Confusion matrices reveal class-level errors

Class-level analysis helps identify visually similar categories that the model finds difficult to distinguish.

---

# 🛠️ Technologies Used

```text
Python
TensorFlow
Keras
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Jupyter Notebook
```

---

# 📁 Project Files

```text
DL_PR4A_Project_A/
│
├── DL_PR4A.ipynb
├── README.md
├── requirements.txt
├── CIFAR10_Dataset_Info.csv
├── cifar10_labels.txt
└── screenshots/
```

Main submission notebook:

```text
DL_PR4A.ipynb
```

---

# 🚀 Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Learning-rate optimization
- Batch normalization
- Dropout optimization
- Deeper CNN architectures
- ResNet-based architectures
- EfficientNet
- Advanced data augmentation
- Additional pretrained models
- Ensemble learning
- Grad-CAM visualization
- Model deployment using Flask/FastAPI
- Web-based image classification
- Mobile deployment

---

# 🎥 Presentation / Viva

The project can be explained in a **5–10 minute presentation** covering:

### Introduction

- Project objective
- CIFAR-10 dataset
- CNN motivation

### A1

- Dataset loading
- EDA
- Normalization

### A2

- Convolution
- Padding
- Stride
- Feature maps

### A3

- Pooling
- Flatten
- LeNet-5
- Custom CNN

### A4

- Backpropagation
- Loss function
- Adam optimizer
- Training
- Evaluation

### A5

- Data augmentation
- Augmented images
- Augmented model comparison

### A6

- Transfer learning
- MobileNetV2
- Feature extraction
- Fine-tuning
- Residual connections

### Final

- Model comparison
- Generated results
- Key findings
- Conclusion

---

# ✅ Conclusion

This project demonstrates a complete **Convolutional Neural Network workflow for CIFAR-10 image classification**.

The work progresses from dataset loading and exploratory analysis to the fundamental components of CNNs:

- Convolution
- Padding
- Stride
- Pooling
- Flattening
- CNN architecture
- Backpropagation
- Model training
- Data augmentation
- Transfer learning
- Fine-tuning
- Residual connections

The project provides practical understanding of how CNNs learn hierarchical visual representations.

It also demonstrates the difference between **training a CNN from scratch**, improving it through **data augmentation**, and using **transfer learning with a pretrained MobileNetV2 model**.

The final numerical performance values should be taken directly from the executed notebook outputs rather than manually entered.

---

# 👩‍💻 Author

**Sanika Kale**

MCA Student | Data Analytics & Machine Learning

**Red & White Skill Education**
