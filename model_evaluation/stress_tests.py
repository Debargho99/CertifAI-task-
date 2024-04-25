
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from PIL import ImageFilter
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Parameters
height, width = 224, 224
batch_size = 64

# Define the transformations for stress testing
def add_random_noise(image):
    noise_factor = 0.1
    noise = np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    noisy_image = image + noise_factor * noise
    noisy_image = np.clip(noisy_image, 0, 1)  # Ensure the pixel values are valid
    return noisy_image

def add_blur(image):
    image = np.array(image * 255, dtype=np.uint8)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 4)
    blurred_image = blurred_image / 255.0  # Rescale back to [0, 1]
    return blurred_image

def apply_advanced_augmentations(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_crop(image, size=[height, width, 3])
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image.numpy()


# Stress testing function
def stress_test_model(model, data_generator):
    for batch in data_generator:
        images, labels = batch
        # Apply transformations
        noisy_images = np.array([add_random_noise(img) for img in images])
        blurred_images = np.array([add_blur(img) for img in images])
        augmented_images = np.array([apply_advanced_augmentations(img) for img in images])

        # Test model on original, noisy, blurred, and augmented images
        for test_images, description in zip([images, noisy_images, blurred_images, augmented_images], 
                                           ['Original', 'Noisy', 'Blurred', 'Augmented']):
            preds = model.predict(test_images)
            print(f"{description} images predictions:")
            # No decode_predictions since we have custom classes
            top_preds = np.argmax(preds, axis=1)
            print("Top class predictions:", top_preds)
        break  # Remove this to test on the entire dataset


def save_confusion_matrices(model, data_generator, output_dir):
    transformations = [
        (identity, 'Original'),
        (add_random_noise, 'Noisy'),
        (add_blur, 'Blurred'),
        (apply_advanced_augmentations, 'Augmented')
    ]

    plt.figure(figsize=(20, 5))  # Set figure size large enough to hold all subplots
    for i, (func, label) in enumerate(transformations, 1):
        true_labels = []
        pred_labels = []

        for batch in data_generator:
            images, labels = batch
            transformed_images = np.array([func(img) for img in images])
            preds = model.predict(transformed_images)
            pred_classes = np.argmax(preds, axis=1)

            true_labels.extend(labels)
            pred_labels.extend(pred_classes)
            
            # Limit to a single batch for demonstration; adjust as necessary
            break

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        # Plotting
        ax = plt.subplot(1, 4, i)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'Confusion Matrix - {label}')
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/stress_test_cm.png")









def perform_stress_test(model, data_generator, output_dir):
    save_confusion_matrices(model, data_generator, output_dir)