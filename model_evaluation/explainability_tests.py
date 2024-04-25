import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

def explain_with_lime(model, img_path, output_dir, top_labels=5, hide_color=0, num_samples=1000):
    explainer = lime_image.LimeImageExplainer()

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image

    explanation = explainer.explain_instance(
        img_array[0].astype('double'), 
        lambda x: model.predict(np.expand_dims(x, axis=0)),
        top_labels=top_labels,
        hide_color=hide_color,
        num_samples=num_samples
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    ax.title.set_text('Explanation for class: ' + str(explanation.top_labels[0]))
    ax.axis('off')

    # Save the explanation image
    fig.savefig(f"{output_dir}/lime_explanation_{img_path.split('/')[-1]}")
    plt.close(fig)  # Close the figure after saving