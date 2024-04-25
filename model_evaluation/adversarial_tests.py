# adversarial_tests.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import preprocess_input

def create_adversarial_pattern(input_image, input_label, model):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(input_label, prediction)

    gradient = tape.gradient(loss, input_image)
    signed_grad = tf.sign(gradient)
    return signed_grad

def generate_and_test_adversarial(model, data_generator, output_dir, epsilon=0.01):
    images, labels = next(data_generator)
    adversarial_examples = []
    correct = 0

    for i, (image, label) in enumerate(zip(images, labels)):
        image = preprocess_input(image)  # Preprocess the image
        image = tf.expand_dims(image, axis=0)
        label = tf.expand_dims(label, axis=0)
        perturbation = create_adversarial_pattern(image, label, model)
        adversarial_image = image + epsilon * perturbation
        adversarial_examples.append(adversarial_image)

        # Predict and calculate accuracy
        pred = model.predict(adversarial_image)
        if np.argmax(pred) == label:
            correct += 1

        # Save the adversarial image
        plt.imsave(f"{output_dir}/adv_image_{i}.png", adversarial_image.numpy().squeeze())

    accuracy = correct / len(labels)
    print(f"Accuracy on adversarial examples: {accuracy * 100:.2f}%")