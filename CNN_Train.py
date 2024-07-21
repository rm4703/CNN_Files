import os

# Path to your dataset
dataset_path = "letters"

# Get the list of class (folder) names
class_names = os.listdir(dataset_path)

print("Found classes:")
for class_name in class_names:
    print(class_name)

# Continue with the rest of your script

import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

# Parameters
img_width, img_height = 224, 224
batch_size = 32
epochs = 100
model_file = 'my_model.h5'  # File to save the trained model

# Function to load and preprocess images
def load_and_preprocess_image(img_path):
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.  # Normalize pixel values to [0, 1]
    return img_array


# Build base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
for layer in base_model.layers:
    layer.trainable = False
# Build a simple CNN model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])
# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])






# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Load and augment training data
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
model.fit(train_generator, epochs=epochs)

# Save the model
model.save(model_file)
print(f"Model saved to {model_file}")
