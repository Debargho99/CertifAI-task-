import argparse
import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model_evaluation.stress_tests import perform_stress_test
from model_evaluation.adversarial_tests import generate_and_test_adversarial
from model_evaluation.explainability_tests import explain_with_lime


# Set up argument parsing
parser = argparse.ArgumentParser(description='Run model evaluation tests')
parser.add_argument('--data_dir', type=str, help='Path to the directory with test dataset')
parser.add_argument('--output_dir', type=str, help='Path to the output directory to save results')
parser.add_argument('--image_paths', nargs='+', help='Paths to images for LIME interpretability tests')

# Parse arguments
args = parser.parse_args()

# Make sure output directory exists or create it
os.makedirs(args.output_dir, exist_ok=True)

# Clear any previous session and setup
tf.keras.backend.clear_session()

# Parameters
height, width = 224, 224
batch_size = 64

# Load the base VGG16 model without the fully connected layers
base_model = tf.keras.applications.VGG16(
    weights='imagenet', 
    include_top=False,
    input_shape=(height, width, 3)
)
base_model.trainable = False

# Build the custom model
model_vgg16 = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(3, activation='softmax')
])

model_vgg16.compile(loss='sparse_categorical_crossentropy', 
              optimizer=tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

weights_path = 'C:/Users/ASUS/Desktop/vgg16_best.h5'  # Adjust this path as needed
model_vgg16.load_weights(weights_path) 

datagen = ImageDataGenerator(rescale=1./255.)
    
test_generator = datagen.flow_from_directory(
        args.data_dir,
        batch_size=batch_size,
        shuffle=True,
        seed=42,
        class_mode='sparse',  # 'binary' changed to 'sparse' for multi-class
        target_size=(height, width),
        classes={'Normal': 0, 'Viral Pneumonia': 1, 'Covid': 2}
    )


# Run stress test and save outputs
perform_stress_test(model_vgg16, test_generator, args.output_dir)

# Run adversarial test and save outputs
generate_and_test_adversarial(model_vgg16, test_generator, args.output_dir)

# If image paths for LIME are provided, run interpretability tests and save outputs
if args.image_paths:
    for image_path in args.image_paths:
        explain_with_lime(model_vgg16, image_path, args.output_dir)
