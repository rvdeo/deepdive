import tensorflow as tf
import pathlib
import PIL
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def basic_data_read():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
    data_dir = pathlib.Path(archive).with_suffix('')

    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[0]))

    roses = list(data_dir.glob('roses/*'))
    PIL.Image.open(str(roses[1]))

    batch_size = 32
    img_height = 180
    img_width = 180

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    plt.show()

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    print(np.min(first_image), np.max(first_image))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    num_classes = 5
    
    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )
    
#basic_data_read()


    
IMG_SIZE = 250

def read_images_in_folder(folder_path):
    image_list = []
    labels = []
    
  
    
    

    resize_and_rescale = tf.keras.Sequential([layers.Resizing(IMG_SIZE, IMG_SIZE),layers.Rescaling(1./255)])

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                try:
                    image = tf.keras.preprocessing.image.load_img(image_path,target_size=[IMG_SIZE,IMG_SIZE])
                    #image = resize_and_rescale(image)
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    image_list.append(image)
                    
                    label = os.path.basename(root)  # Assuming folder names are the labels
                    labels.append(label)
                except (IOError, OSError):
                    print(f"Error opening image: {image_path}")
    return image_list, labels


def save_images_to_folder(images, labels, folder_path):
    i = 1
    for image, label in zip(images, labels):
        label_folder = os.path.join(folder_path, label)
        os.makedirs(label_folder, exist_ok=True)
        
        image_filename = os.path.join(label_folder, f"{label}_{i}_image.jpg")
        tf.keras.preprocessing.image.save_img(image_filename, image)
        i+=1


def read_preprocess_data():

    folder_path = "Dataset_generated"
    images, labels = read_images_in_folder(folder_path)
    print(f"Total number of images: {len(images)}")
    print(f"Total number of labels: {len(labels)}")
    distinct_labels = list(set(labels))
    
    print(distinct_labels)
    
    
    # Convert the image and label lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Perform train-test-validation split
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.165, random_state=42, stratify=labels)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42, stratify=train_labels)
    
    # Print the shapes of train, validation, and test sets
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Validation images shape:", val_images.shape)
    print("Validation labels shape:", val_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
            
    # Create "data" folder
    data_folder = "preprocessed_data"
    
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
        print(f"Folder '{data_folder}' and its contents have been removed.")
    else:
        print(f"Folder '{data_folder}' does not exist.")
        
    os.makedirs(data_folder, exist_ok=True)
    
    # Save train images to folder
    train_folder = os.path.join(data_folder, "train")
    save_images_to_folder(train_images, train_labels, train_folder)
    
    # Save test images to folder
    test_folder = os.path.join(data_folder, "test")
    save_images_to_folder(test_images, test_labels, test_folder)
    
    # Save validation images to folder
    val_folder = os.path.join(data_folder, "validation")
    save_images_to_folder(val_images, val_labels, val_folder)

    return train_labels, val_labels, test_labels, labels

#read_preprocess_data()



def load_image_data(train_data_dir, test_data_dir, val_data_dir, preprocessing_function,image_size=(IMG_SIZE, IMG_SIZE), batch_size=32):
    # Define image data generator with desired preprocessing and augmentation options
    image_data_generator = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        rescale=1./255,  # Normalize pixel values to [0, 1]
        rotation_range=20,  # Randomly rotate images by up to 20 degrees
        width_shift_range=0.1,  # Randomly shift images horizontally by up to 10% of the width
        height_shift_range=0.1,  # Randomly shift images vertically by up to 10% of the height
        shear_range=0.2,  # Randomly apply shearing transformation
        zoom_range=0.2,  # Randomly zoom in on images
        horizontal_flip=True  # Randomly flip images horizontally
    )

    # Load the train data
    train_data = image_data_generator.flow_from_directory(
        train_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'  # Set to 'binary' if you have binary classification
    )

    # Load the test data
    test_data = image_data_generator.flow_from_directory(
        test_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical'  # Set to 'binary' if you have binary classification
    )

    # Load the validation data
    val_data = image_data_generator.flow_from_directory(
        val_data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical'  # Set to 'binary' if you have binary classification
    )

    # Return the loaded data
    return train_data, test_data, val_data


def plot_images(img, labels,class_names ):
    plt.figure(figsize=[15, 10])
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(img[i])
        plt.title(class_names[np.argmax(labels[i])])
        plt.axis('off')   
    
