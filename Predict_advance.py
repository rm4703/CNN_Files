import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to preprocess and localize text in a single image
def preprocess_and_localize_text(image_path, output_dir, min_area_threshold=1000):
    # Read the image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to create binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Morphological operations to remove noise and improve text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours and localize text
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a list to store valid contours
    valid_contours = []

    # Process each contour
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)

        # Only process contours above the area threshold
        if area > min_area_threshold:
            # Get bounding box coordinates
            (x, y, w, h) = cv2.boundingRect(contour)

            # Append the contour and bounding box coordinates to the list
            valid_contours.append((x, contour, (x, y, w, h)))

    # Sort contours based on x-coordinate (left to right)
    valid_contours.sort(key=lambda x: x[0])

    # Save and process each valid contour
    saved_image_paths = []
    for i, (x, contour, (x, y, w, h)) in enumerate(valid_contours):
        # Crop the bounding box region from the original image
        cropped_img = image[y:y + h, x:x + w]

        # Save the cropped image to the output directory with a sequential filename
        output_path = os.path.join(output_dir, f"localized_text_{i + 1}.jpg")
        cv2.imwrite(output_path, cropped_img)

        # Append the saved image path to the list
        saved_image_paths.append(output_path)

        # Draw rectangle on original image (optional)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display or further process localized image with rectangles (optional)
    cv2.imshow("Localized Text Regions", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return saved_image_paths

# Function to load and preprocess an image for prediction
def load_and_preprocess_image(img_path, img_width=224, img_height=224):
    img = load_img(img_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.  # Normalize pixel values to [0, 1]
    return img_array

# Example usage
image_path = 'testA.jpg'
output_directory = 'temp_files'
saved_image_paths = preprocess_and_localize_text(image_path, output_directory, min_area_threshold=1000)

# Path to the saved model
saved_model_path = 'my_model.h5'

# Load the saved model
model = load_model(saved_model_path)

# Create a data generator to extract class labels
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'dataset/letters',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical')

# Get the class labels
class_indices = train_generator.class_indices
labels = {v: k for k, v in class_indices.items()}

# Initialize a list to store predicted labels
predicted_labels = []

# Example of predicting each localized text image
for img_path_to_predict in saved_image_paths:
    if os.path.isfile(img_path_to_predict):
        img = load_and_preprocess_image(img_path_to_predict)
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction)  # Get the index of the highest probability
        predicted_label = labels[predicted_class_index]  # Get the corresponding label
        predicted_labels.append(predicted_label)
    else:
        print(f"Error: '{img_path_to_predict}' is not a valid file path.")

# Join the predicted labels into a single string
predicted_text = ''.join(predicted_labels)
print(f"Predicted text: {predicted_text}")
