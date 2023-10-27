import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import load_model

def build_model():

    model = Sequential()
    
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())

    return model
    
def get_data():
    data_dir = 'data' 
    
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                # os.remove(image_path)
                
                
    data = tf.keras.utils.image_dataset_from_directory('data')
    
    data_iterator = data.as_numpy_iterator()            
    batch = data_iterator.next()
    
 
    
    fig, ax = plt.subplots(ncols=len(batch[0]), figsize=(30,5))
  
    for idx, img in enumerate(batch[0][:]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(batch[1][idx]) 
            ax[idx].set_axis_off()

      
    plt.show()
    
               
    data = data.map(lambda x,y: (x/255, y))
    
   
    
    data.as_numpy_iterator().next()

    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    
    return train,val, test


def train():
    
    
    
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus: 
        tf.config.experimental.set_memory_growth(gpu, True)
        
    tf.config.list_physical_devices('GPU')
    
    data_dir = 'data' 
    
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                # os.remove(image_path)
                
                
    data = tf.keras.utils.image_dataset_from_directory('data')
    
    data_iterator = data.as_numpy_iterator()            
    batch = data_iterator.next()
    
 
    
    fig, ax = plt.subplots(ncols=len(batch[0]), figsize=(30,5))
  
    for idx, img in enumerate(batch[0][:]):
            ax[idx].imshow(img.astype(int))
            ax[idx].title.set_text(batch[1][idx]) 
            ax[idx].set_axis_off()

      
    plt.show()
    
               
    data = data.map(lambda x,y: (x/255, y))
    
   
    
    data.as_numpy_iterator().next()

    train_size = int(len(data)*.7)
    val_size = int(len(data)*.2)
    test_size = int(len(data)*.1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)
    
    print(train_size, val_size, test_size)
       
    print(len(data))
      
    
    
    model = Sequential()
    
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    print(model.summary())
    
    logdir='logs'
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
    
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
    
    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()   
    
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()
    
    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
        
    print(pre.result(), re.result(), acc.result())
    
    img = cv2.imread('sponge.png')
    plt.imshow(img)
    plt.show()
        
    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()

    yhat = model.predict(np.expand_dims(resize/255, 0))
    
    if yhat > 0.5: 
        print(f'Predicted class is prawn')
    else:
        print(f'Predicted class is sponge')
    model.save(os.path.join('models','imageclassifier.h5'))
    
from tkinter import filedialog as fd
def test():
    
    tr,vl,ts = get_data()
    
    print('***************************************\n Test_data\n********************************************************')
    print(ts)
    
    new_model = load_model('models/imageclassifier.h5')
    score = new_model.evaluate(ts, verbose=1)
       
    for i in range(4):
        
        
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])
        filename = fd.askopenfilename()
        img = cv2.imread(filename)
        #plt.imshow(img)
        #plt.show()
        resize = tf.image.resize(img, (256,256))
        plt.imshow(resize.numpy().astype(int))
        plt.show()
        
        yhat = new_model.predict(np.expand_dims(resize/255, 0))
        if yhat > 0.7: 
            print(f'Predicted class is prawn')
        elif yhat >0.3:
            print(f'Predicted class is unknown')
        else:
            print(f'Predicted class is sponge')
    
   

test()
