# Import all the necessary libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the CIFAR-10 dataset and divide it into training and testing sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Check the shape of the datasets
print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

# Normalize the images scaling pixel values to be between 0 and 1
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert class labels to one-hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Display a few random images from the training set
plt.figure(figsize=(10,10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i])
    plt.title(f"Class: {y_train[i].argmax()}")
    plt.axis('off')
plt.show()

# Build a Convolutional Neural Network (CNN) suitable for image classification.

# Model / data parameters
num_classes = 10
input_shape = (32, 32, 3)

# Build the CNN with convolutional blocks and poolings (flatten, dropout and dense at the end to fully connect layers)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(), # Adding batch normalization after each convolutional block can help stabilize and faster the training.
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),  # Add dropout to reduce overfitting

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),
    
        layers.Flatten(),
        layers.Dense(256, activation="relu"), # Add a fully connected layer before the output
        layers.Dropout(0.5), # Increase dropout for the fully connected layer
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0003), metrics=["accuracy"])

# Train the model
history_cnn = model.fit(X_train[:8000], y_train[:8000], batch_size=32, epochs=60, validation_split=0.1)

# Function to plot the training and validation accuracy and loss
def plot_accuracy_and_loss(history, model_name):
    # Plot accuracy
    plt.figure(figsize=(14, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Plot accuracy and loss for the Custom CNN model
plot_accuracy_and_loss(history_cnn, 'CNN Model')

# List of CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Compute and report metrics 
score = model.evaluate(X_test, y_test, verbose=3)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Get predictions for the test set
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
y_true = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Visualize the confusion matrix to understand model performance across different classes.
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for CIFAR-10')
plt.show()

# Save Model as pickle file
import pickle
with open('CNN_Charlie_Dani.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load VGG16 model pre-trained on ImageNet, excluding the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add GlobalAveragePooling2D layer to flatten the output of VGG16
avg = GlobalAveragePooling2D()(base_model.output)

# Add a fully connected layer for classification
output = layers.Dense(num_classes, activation='softmax')(avg)

# Add new fully connected layers on top of VGG16 for CIFAR-10 (10 classes)
combined_model = models.Model(inputs=base_model.input, outputs=output)

# Summary of the model
combined_model.summary()

# Compile the model
combined_model.compile(optimizer=Adam(learning_rate=0.0003), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model using the CIFAR-10 dataset
history_vgg16 = combined_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

# Unfreeze the last few layers of the base model for fine-tuning
for layer in base_model.layers[-4:]:
    layer.trainable = True

# Recompile and fine-tune the model
combined_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training
history_vgg16 = combined_model.fit(X_train[:8000], y_train[:8000], epochs=10, batch_size=16, validation_split=0.1)

# Evaluate on the test data
score = combined_model.evaluate(X_test, y_test, verbose=2)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Plot accuracy and loss for the Custom CNN model
plot_accuracy_and_loss(history_vgg16, 'VGG16 Transfer Learning Model')

# Get predictions for the test set
y_pred = combined_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Classification report (Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for CIFAR-10 after Fine-Tuning VGG16')
plt.show()

# Save Model as pickle file
import pickle
with open('VGG16_Charlie_Dani.pkl', 'wb') as f:
    pickle.dump(combined_model, f)