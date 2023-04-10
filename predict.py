from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import PIL
import matplotlib

matplotlib.rcParams['figure.figsize'] = [10, 20]


def predict(img_path, weights_file=None):

    if weights_file == None:
        model = load_model("model.hdf5")
    else:
        model = load_model(weights_file)

    img = PIL.Image.open(img_path)
    img = img.resize((256, 256))
    img = np.asarray(img, dtype=np.float32)
    img = img / 255
    copy_img = img.reshape(-1, 256, 256, 3)
    predict = model.predict(copy_img)

    # Create a subplot with 1 rows, 2 column, and share x-axis
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first figure on the first subplot
    axs[0].imshow(img)

    labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR',
              3: 'Proliferate_DR', 4: 'Severe'}

    axs[0].set_xlabel("Prediction: " + labels[np.argmax(predict)])

    # Plot the second figure on the second subplot
    axs[1].bar(list(labels.values()), predict[0].tolist(),
               color='blue', width=0.4)
    # Enable label wrapping for x-axis labels in axs[1]
    axs[1].set_xticklabels(list(labels.values()), wrap=True)

    # Set additional properties for x-axis tick labels in axs[1]
    # Set labels and title for the combined figure

    # Set x-axis label for the second subplot

    # Display the combined figure
    plt.show()
