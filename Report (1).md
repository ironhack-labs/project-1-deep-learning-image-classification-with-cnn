
# Image Classification Project: CIFAR-10

## **Introduction**
This project focuses on image classification using two approaches:
1. A custom Convolutional Neural Network (CNN) designed and trained from scratch.
2. Transfer Learning using a pre-trained VGG16 model fine-tuned on the CIFAR-10 dataset.

---

## **Methodology**

### **1. Data Preprocessing**
- **Normalization**: Images were normalized to the range [-1, 1].
- **One-Hot Encoding**: Labels were converted into one-hot encoded vectors.
- **Data Augmentation**: Techniques like random rotation, zoom, and flips were applied to increase dataset diversity.

### **2. Model Architectures**
#### **Custom CNN**
- Includes convolutional layers, max-pooling, dropout, and dense layers.
- Trained for 20 epochs with EarlyStopping.

#### **VGG16 (Transfer Learning)**
- Pre-trained on ImageNet.
- Fully connected layers were replaced with a custom head for CIFAR-10 classification.

---

## **Training Details**
| Parameter       | Value                |
|-----------------|----------------------|
| Batch Size      | 64                   |
| Learning Rate   | 0.0001               |
| Optimizer       | Adam                 |
| Early Stopping  | Patience = 5 epochs  |

---

## **Results**

### **Custom CNN**
- Test Accuracy: **71%**
- Confusion Matrix: Observed confusions in `bird` and `cat`.

### **VGG16**
- Test Accuracy: **62%**
- Confusion Matrix: Similar patterns of confusion, though accuracy was lower.

---

## **Conclusions**
- The custom CNN performed better than VGG16 for this dataset, likely due to its simplicity and lower resolution of CIFAR-10 images.
- Transfer Learning could be further optimized by unfreezing more layers or trying other architectures like ResNet.
- Data augmentation and careful pre-processing significantly improved model performance.

