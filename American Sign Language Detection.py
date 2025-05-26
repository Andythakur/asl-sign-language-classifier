import os
from tensorflow.keras.models import load_model

MODEL_PATH = "asl_model.h5"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from tensorflow.keras.preprocessing import image

import gradio as gr

# Image and batch configuration
IMG_SIZE = (64, 64)
BATCH_SIZE = 32

# Set the correct path to the training directory (contains folders A-Z, SPACE, DELETE, NOTHING)
train_dir = '/Users/andy/Downloads/archive/asl_alphabet_train/asl_alphabet_train'

# Use 80% for training and 20% for validation
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Define CNN model
from tensorflow.keras import Input

if os.path.exists(MODEL_PATH):
    print("✅ Loading existing model...")
    model = load_model(MODEL_PATH)
else:
    print("🚀 Training new model...")
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(29, activation='softmax')  # 29 classes
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=val_generator
    )

    model.save(MODEL_PATH)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {test_acc * 100:.2f}%")

# Prediction on a single image
img_path = '/Users/andy/Downloads/archive/asl_alphabet_train/asl_alphabet_train/A/A1.jpg'
img = image.load_img(img_path, target_size=IMG_SIZE)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict the class
prediction = model.predict(img_array)
predicted_class = list(train_generator.class_indices.keys())[np.argmax(prediction)]
print(f"Predicted Sign: {predicted_class}")

# Gradio UI
def predict_image(img):
    if img is None:
        return "No image uploaded."

    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = list(train_generator.class_indices.keys())[np.argmax(prediction)]
    return predicted_class

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload ASL Image"),
    outputs=gr.Text(label="Predicted ASL Sign"),
    title="ASL Sign Language Classifier",
    description="Upload an image of an ASL hand sign to detect the corresponding letter or command (SPACE, DELETE, NOTHING)."
).launch()