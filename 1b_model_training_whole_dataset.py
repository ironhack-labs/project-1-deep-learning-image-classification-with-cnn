# %% [markdown]
# # 0. Libraries

# %%
import pickle
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications import InceptionV3
from tensorflow.image import resize


# %% [markdown]
# # 1. Data Preprocessing

# %%
# Data loading
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

batch_1 = unpickle("data/data_batch_1")
batch_2 = unpickle("data/data_batch_2")
batch_3 = unpickle("data/data_batch_3")
batch_4 = unpickle("data/data_batch_4")
batch_5 = unpickle("data/data_batch_5")
test_batch = unpickle("data/test_batch")
label_names = unpickle("data/batches.meta")

# %%
# Turn the labels lists into np.arrays()
batch_1[b'labels'] = np.asarray(batch_1[b'labels'])
batch_2[b'labels'] = np.asarray(batch_2[b'labels'])
batch_3[b'labels'] = np.asarray(batch_3[b'labels'])
batch_4[b'labels'] = np.asarray(batch_4[b'labels'])
batch_5[b'labels'] = np.asarray(batch_5[b'labels'])
test_batch[b'labels'] = np.asarray(test_batch[b'labels'])

# %%
# Reshape every image from (3073,) to (32,32,3)so we can see it with plt.imshow()
def reshape_transpose(batch):
    images = batch[b"data"].reshape(10000, 3, 32, 32) # Because of how np.reshape works, this returns an array with np.shape=(10000,3,32,32)
    images = images.transpose(0,2,3,1) # We transpose it so it has the correct np.shape=(10000,32,32,3) for plt.imshow()
    return images

images1 = reshape_transpose(batch_1)
images2 = reshape_transpose(batch_2)
images3 = reshape_transpose(batch_3)
images4 = reshape_transpose(batch_4)
images5 = reshape_transpose(batch_5)
test_images = reshape_transpose(test_batch)

# %% [markdown]
# # 2. Model Architecture

# %%
# Prepare the data
num_classes = len(label_names[b'label_names']) # Length of the label_names list --> 10

# We don't need to do a train_test_split because the data is already split
X_train = np.concatenate((images1, images2, images3, images4, images5))
y_train = np.concatenate((batch_1[b'labels'], batch_2[b'labels'], batch_3[b'labels'], batch_4[b'labels'], batch_5[b'labels']))
X_test = test_images
y_test = test_batch[b'labels']

# Convert labels to categorical. This way, every int [0:10] is a class and it won't be treated as continous
y_train = to_categorical(y_train, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# Scale images to the [0, 1] range
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# Model parameters
input_shape = X_train[0].shape # Shape of any image from any of the batches --> (32, 32, 3)

# Model Architecture
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),

        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        # layers.Dropout(0.4),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.4),

        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        # layers.Dropout(0.4),

        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.4),
        

        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.4),
        

        layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.4),
        

        layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Dropout(0.4),
        

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        layers.Dense(128, activation="relu"),
        # layers.Dropout(0.4),


        layers.Flatten(),
        
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy", "precision", "recall"])
# model.compile(loss="categorical_crossentropy", optimizer=SGD(learning_rate=0.001, momentum=0.9, nesterov=True), metrics=["accuracy", "precision", "recall"])

# %%
# Create a checkpoint to save the model while it's training
checkpoint = ModelCheckpoint(
    filepath='model_epoch_{epoch:02d}_val_acc_{val_accuracy:.4f}.keras',  # Path to save the model file
    monitor='val_accuracy',          # Metric to monitor (you can also use 'val_accuracy' or other metrics)
    save_best_only=True,             # Save only the best model (based on monitored metric)
    mode='max',                      # Minimize the monitored metric (for 'val_loss', use 'min'; for accuracy, use 'max')
    save_weights_only=False,         # Whether to save the whole model or just the weights
    verbose=0                        # Display info when saving
)

# %%
batch_size = 128
epochs = 30
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[checkpoint])

#%%
pickle.dump(model, open(f"trained_model_whole_dataset.pkl", "wb"))
# %% [markdown]
# # 5. Transfer Learning

# %%
# Preprocess the data to adapt it to the Inception model
X_train_inception= resize(X_train, (75,75))
X_test_inception = resize(X_test, (75,75))

# Model parameters
input_shape_inception = X_train_inception[0].shape

# %%
# Extract the Inception pre-trained model
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=input_shape_inception) # weights='imagenet' loads the model trained on the ImageNet dataset. include_top=False stops the model from loading the first layer

# %%
base_model.trainable = False # Freeze the layers of the base model

avg = GlobalAveragePooling2D()(base_model.output) # This layer transforms 2D into 1D, or from a Convolution layer to a Dense layer
output = Dense(num_classes, activation="softmax")(avg) # This layer classifies the input into one of n_classes categories

combined_model = Model(inputs=base_model.input, outputs=output) # Combining these layers to create the combined model of Inception with a new top layer



# %%
# Train the new top layer with a high learning rate
combined_model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.2), metrics=["accuracy", "precision", "recall"])

batch_size_inception = 128
epochs_inception = 30
history = combined_model.fit(X_train_inception, y_train, batch_size=batch_size_inception, epochs=epochs_inception, validation_split=0.1, callbacks=[checkpoint])

# %%
# Fine-tune the whole model
base_model.trainable = True # Unfreeze the layers of the base model

for layer in base_model.layers[:150]: # Freeze only some layers of the base model to avoid overfitting
    layer.trainable = False

combined_model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.0001), metrics=["accuracy", "precision", "recall"])

history = combined_model.fit(X_train_inception, y_train, batch_size=batch_size_inception, epochs=epochs_inception, validation_split=0.1)

# %%
# Save the combined model
pickle.dump(combined_model, open("combined_model_whole_dataset.pkl", "wb"))


