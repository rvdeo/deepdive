
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras.optimizers.legacy import Adam

#import pretrained models
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101
from keras.applications.resnet import ResNet152
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
#from keras.applications.inceptionresnetv2 import InceptionResNetV2

from tensorflow.keras.models import load_model
from utils import *
from sklearn.metrics import classification_report
import seaborn as sns
import os
import time
import csv
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import RocCurveDisplay
from itertools import cycle
import random

tf.get_logger().setLevel('INFO')

def create_results_folder(directory):
    
    if not os.path.exists(directory):
      
        # if the demo_folder directory is not present 
        # then create it.
        os.makedirs(directory)
    
    
    # Get all existing folder names in the directory
    folder_names = next(os.walk(directory))[1]

    # Find the highest counter value
    max_counter = 0
    for folder_name in folder_names:
        if folder_name.startswith("results"):
            try:
                counter = int(folder_name.split("_")[1])
                max_counter = max(max_counter, counter)
            except IndexError:
                continue
            except ValueError:
                continue

    # Increment the counter and create the new results folder
    new_counter = max_counter + 1
    new_folder_name = "results_{:02d}".format(new_counter)
    new_folder_path = os.path.join(directory, new_folder_name)
    os.makedirs(new_folder_path)

    return new_folder_path

def generate_random_colors(n):
    # Generate n random colors
    colors = []
    for _ in range(n):
        r = random.random()
        g = random.random()
        b = random.random()
        colors.append((r, g, b))

    return colors


def plot_train_accuracy_loss(acc,val_acc, loss, val_loss,results_folder):
    # plot results
    # accuracy
    plt.figure(figsize=(10, 16))
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.grid'] = True
    plt.rcParams['figure.facecolor'] = 'white'
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.title(f'\nTraining and Validation Accuracy. \nTrain Accuracy:{str(acc[-1])}\nValidation Accuracy: {str(val_acc[-1])}')
    
    # loss
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title(f'Training and Validation Loss. \nTrain Loss:{str(loss[-1])}\nValidation Loss: {str(val_loss[-1])}')
    plt.xlabel('epoch')
    plt.tight_layout(pad=3.0)
    plt.savefig(results_folder+"_training_perf.png")
    plt.show()

def plot_model_accuracies(model_names,accuracies,results_folder):
    """
    Plot model accuracies.

    Args:
        model_accuracy (dict): Dictionary containing model names as keys and their accuracies as values.

    Returns:
        None
    """


    # Plot the accuracies
    plt.bar(model_names, accuracies)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracies')
    plt.xticks(rotation=45)
    plt.savefig(results_folder+"/model_accuracies.png")
    plt.show()
    
def main():
    
    print(tf.config.list_physical_devices('GPU'))
    #return
    
    print("Starting Experiments")
    # Specify the directory path
    directory_path = "results"
    
    # Create the new results folder
    results_folder = create_results_folder(directory_path)
    
    # Print the path of the new results folder
    print("New results folder created:", results_folder)
    
  
        
   
    train_model = True
    
    if train_model==True:
        
        
        base_model = None
        preprocessing_function = None
        
        test_acc_list = []
        test_per_class_summary = []
        model_list = ['InceptionResNetV2']#, 'VGG16', 'VGG19']# ,'ResNet50', 'ResNet101','ResNet152' ,'InceptionV3' , 'DenseNet121','DenseNet169',  'DenseNet201']
        
        for model_name in model_list:
            
            
            if model_name == 'InceptionResNetV2':
                # load the InceptionResNetV2 architecture with imagenet weights as base
                #from keras.applications.inceptionresnetv2 import preprocess_input
                #preprocessing_function = preprocess_input
                base_model = tf.keras.applications.InceptionResNetV2( include_top=False, weights='imagenet', input_shape=(IMG_SIZE,IMG_SIZE,3))
            
            elif model_name == 'VGG16':
                from keras.applications.vgg16 import preprocess_input 
                preprocessing_function = preprocess_input
                base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
            
            elif model_name == "VGG19":
                from keras.applications.vgg19 import preprocess_input
                preprocessing_function = preprocess_input
                base_model = VGG19(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
            
            elif model_name == "ResNet50":
                from keras.applications.resnet50 import preprocess_input
                preprocessing_function = preprocess_input
                base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
           
            elif model_name == "ResNet101":
                from keras.applications.resnet101 import preprocess_input
                preprocessing_function = preprocess_input
                base_model = ResNet101(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
                
            elif model_name == "ResNet152":
                from keras.applications.resnet152 import preprocess_input
                preprocessing_function = preprocess_input
                base_model = ResNet152(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
                
            elif model_name == "InceptionV3":
                from keras.applications.inception_v3 import preprocess_input
                preprocessing_function = preprocess_input
                base_model = InceptionV3(weights='imagenet', include_top=False,  input_shape=(IMG_SIZE,IMG_SIZE,3))
                
            elif model_name == "DenseNet121":
                from keras.applications.densenet import preprocess_input
                preprocessing_function = preprocess_input
                base_model = DenseNet121(weights='imagenet', include_top=False,  input_shape=(IMG_SIZE,IMG_SIZE,3))
                
            elif model_name == "DenseNet169":
                from keras.applications.densenet import preprocess_input
                preprocessing_function = preprocess_input
                base_model = DenseNet169(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
            
            elif model_name == "DenseNet201":
                from keras.applications.densenet import preprocess_input
                preprocessing_function = preprocess_input
                base_model = DenseNet201(weights='imagenet', include_top=False, input_shape=(IMG_SIZE,IMG_SIZE,3))
                
        
            base_model.trainable=False
            # For freezing the layer we make use of layer.trainable = False
            # means that its internal state will not change during training.
            # model's trainable weights will not be updated during fit(),
            # and also its state updates will not run.
            
            # load data for training
            train_data_dir = 'preprocessed_data/train'
            test_data_dir = 'preprocessed_data/test'
            val_data_dir = 'preprocessed_data/validation'
    
            train_ds, test_ds, validation_ds = load_image_data(train_data_dir, test_data_dir, val_data_dir,preprocessing_function)
            
            x,y = next(train_ds)
            print(x.shape) # input shape of one record is (IMG_SIZE,IMG_SIZE,3) , 32: is the batch size
            a = train_ds.class_indices
            class_names = list(a.keys())  # storing class/breed names in a list 
            plot_images(x,y,class_names)
            
            # Print the number of classes and the class labels
            num_classes = train_ds.num_classes
            print("Number of classes:", num_classes)
            print("Class labels:", class_names)
            
            
            #create transfer learning model
            model = tf.keras.Sequential([
                    base_model,  
                    tf.keras.layers.BatchNormalization(renorm=True),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dense(num_classes, activation='softmax')
                ])
            
            model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
            # categorical cross entropy is taken since its used as a loss function for
            # multi-class classification problems where there are two or more output labels.
            # using Adam optimizer for better performance
            
            
            model.summary()
            
            EPOCHS = 1
            batch_size=32
            STEP_SIZE_TRAIN = train_ds.n//train_ds.batch_size
            STEP_SIZE_VALID = validation_ds.n//validation_ds.batch_size
            
            
            early = tf.keras.callbacks.EarlyStopping( patience=10,
                                                      min_delta=0.001,
                                                      restore_best_weights=True)
            # early stopping call back
            
            
            # fit model
            history = model.fit(train_ds,
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=validation_ds,
                                validation_steps=STEP_SIZE_VALID,
                                epochs=EPOCHS,
                                callbacks=[early])
            
                          
            model.save(results_folder+"/"+model_name+"_Model.h5")
            
            
            # store results
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            
            
             
            plot_train_accuracy_loss(acc, val_acc, loss, val_loss, results_folder+"/"+model_name)
            
            
            accuracy_score = model.evaluate(test_ds)
            print(accuracy_score)
            print("Accuracy: {:.4f}%".format(accuracy_score[1] * 100))
            print("Loss: ",accuracy_score[0])
            
            test_acc_list.append(accuracy_score[1] * 100)
            
            summary_results = test_model(train_ds, test_ds, results_folder+"/"+model_name ,train_model)
            
            mod_details = [model_name, EPOCHS, accuracy_score[1] * 100]
            
            
            
            append_dict_list_to_csv(summary_results, mod_details, class_names ,results_folder+"/model_summaries.csv")
           
            
        
        plot_model_accuracies(model_list,test_acc_list,results_folder)
            
    else:
        
        # load data for training
        train_data_dir = 'preprocessed_data/train'
        test_data_dir = 'preprocessed_data/test'
        val_data_dir = 'preprocessed_data/validation'

        preprocessing_function = None
        
        train_ds, test_ds, validation_ds = load_image_data(train_data_dir, test_data_dir, val_data_dir,preprocessing_function)
        
        test_model(train_ds, test_ds, results_folder,train_model)
        


def append_dict_list_to_csv(dict_list, model_name, class_names, filename):
    """
    Append a list of dictionaries to an existing CSV file.

    Args:
        dict_list (list): List of dictionaries.
        filename (str): Name of the CSV file to append to.

    Returns:
        None
    """
    #write header row for each model
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write the line of data
        writer.writerow(model_name)
        writer.writerow(class_names)
        
        
    
    # Get the field names from the keys of the first dictionary
    fieldnames = list(dict_list[0].keys())

    # Open the CSV file in append mode
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       
        # Write the data rows
        writer.writerows(dict_list)

def test_model(train_ds, test_ds, results_folder,train_model):
    
    model = None
    if train_model==True:
        model = load_model(results_folder+"_Model.h5")
    else:
        model = load_model("results/results_104/InceptionResNetV2_Model.h5")
           
    pred= model.predict(test_ds, verbose=0)
    
    
    
    predicted_class_indices=np.argmax(pred,axis=1)
    labels = (train_ds.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    
    
    test_labels = []
    test_images = []
    for i in range(len(test_ds)):
        batch_images, batch_labels = test_ds[i]
        test_labels.extend(tf.argmax(batch_labels, axis=1).numpy())
        test_images.extend(batch_images)
    
    # Convert test_labels to a numpy array
    test_labels = tf.convert_to_tensor(test_labels)
       
    
    
    print(np.array(predicted_class_indices).shape)
    print(np.array(test_labels).shape)
    
    y_true = np.array(test_labels)
    y_pred = np.array(predicted_class_indices)
    class_labels = list(train_ds.class_indices.keys())
    
    #plot_confusion_matrix(y_true, y_pred, classes=class_labels, results_folder)
    
    plot_confusion_matrix(y_true, y_pred, class_labels,results_folder, normalize=True)
    
    print(classification_report(y_true, y_pred, target_names=class_labels))
    
    #plot samlpe model output
    #prediction_titles = [
     #   title(y_pred, y_true, class_labels, i) for i in range(y_pred.shape[0])
    #]
    
    #plot_gallery(test_images, prediction_titles, IMG_SIZE, IMG_SIZE)
    
    if False:
        x,y = next(test_ds)
        print(x.shape) # input shape of one record is (IMG_SIZE,IMG_SIZE,3) , 32: is the batch size
        a = train_ds.class_indices
        class_names = list(a.keys())  # storing class/breed names in a list 
        model.predict(x,y,verbose=1)
        plot_images(x,y,class_names)
    
    return plot_multiclass_roc(y_true, pred, class_labels, results_folder )    


def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(" ", 1)[-1]
    true_name = target_names[y_test[i]].rsplit(" ", 1)[-1]
    return "predicted: %s\ntrue:      %s" % (pred_name, true_name)



def plot_multiclass_roc(ground_truth, scores, class_labels,results_folder):
    # Convert ground truth labels to one-hot encoded form
    n_classes = len(class_labels)
    ground_truth_binary = label_binarize(ground_truth, classes=range(n_classes))


    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(ground_truth_binary[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    #plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = cycle(generate_random_colors(n_classes))  # Customize colors if needed
    '''
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label='{0} (AUC = {1:.2f})'.format(class_labels[i], roc_auc[i]))
    '''  
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            ground_truth_binary[:, class_id],
            scores[:, class_id],
            name=f"{class_labels[class_id]}",
            color=color,
            ax=ax,
            
        )

    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right',fontsize='small')
    plt.savefig(results_folder+"_roc_auc.png")
    plt.show()
    
    return [fpr,tpr,roc_auc]


def plot_confusion_matrix(y_true, y_pred, classes, results_folder, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    Plot a confusion matrix.

    Args:
        y_true (array-like): Array of true labels.
        y_pred (array-like): Array of predicted labels.
        classes (list): List of class labels.
        normalize (bool, optional): Whether to normalize the confusion matrix. Default is False.
        title (str, optional): Title of the plot. Default is 'Confusion Matrix'.
        cmap (matplotlib colormap, optional): Colormap for the plot. Default is plt.cm.Blues.

    Returns:
        None
    """
    # Compute confusion matrix
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true][pred] += 1

    # Normalize confusion matrix if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        fmt = '.2f'
    else:
        fmt = 'd'

    thresh = cm.max() / 2.
    for i in range(n_classes):
        for j in range(n_classes):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(results_folder+"_confusion_matrix.png")
    plt.show()
    

    

     
if __name__ ==  "__main__" :
    main()