from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow import expand_dims
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import os

def predict_test_set(path, normal=True):
        '''Used for predicting images in the test set'''
        img = []
        path_format = path + '{0}'

        for filepath in os.listdir(path):
                x = image.load_img(path_format.format(filepath), target_size=(256, 256),
                        color_mode="grayscale")
                x = image.img_to_array(x)
                x = expand_dims(x, 0)
                img.append(x)
        
        preds = []
        for i in img:
                predictions = model.predict(i)
                preds.append(predictions[0][0])
                print(predictions)
        
        count = 0
        for pred in preds:
                if normal:
                        if pred < 0.5:
                                count += 1
                else:
                        if pred >= 0.5:
                                count += 1

        return count
 

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to the input dataset")
ap.add_argument("-m", "--model", required=True,
        help="path to save the model")
ap.add_argument("-o", "--output", required=True,
        help="path to the output plot")
args = vars(ap.parse_args())

# loading the dataset from disk
print("[INFO] loading the dataset...")
train_dataset = image_dataset_from_directory(
        args["dataset"], labels='inferred', label_mode='binary',
        color_mode="grayscale", image_size=(256, 256), subset="training",
        seed=7, validation_split=0.2)

validation_dataset = image_dataset_from_directory(
        args["dataset"], labels='inferred', label_mode='binary',
        color_mode="grayscale", image_size=(256, 256), subset="validation",
        seed=7, validation_split=0.2)

test_dataset = image_dataset_from_directory(
        "xray_dataset/test/", labels='inferred', label_mode='binary',
        color_mode='grayscale', image_size=(256, 256), seed=7)

# building the model. we also apply standardization within the model
model = Sequential([
    # standardizing the input matrix to the range [0, 1]
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 1)),

    # the first (conv => relu => pool) * 3 layer set
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # the second (conv => relu => pool) * 3 layer set
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Dropout(0.25),

    # first (and only) set of FC => relu layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    #layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    # 1 output neuron, with sigmoid
    layers.Dense(1, activation='sigmoid'),
])

# compiling the model
#opt = SGD(lr=0.05, decay=0.05, momentum=0.9, nesterov=True)
opt = Adam(lr=0.0001)
model.compile(optimizer=opt,
        loss='binary_crossentropy',
        metrics=["accuracy"])

model.summary()

# training the model
print("[INFO] training the network...")
epochs = 15
H = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        batch_size=64,
        epochs=epochs,
        verbose=1,
    )

# loading the pre-trained model
#print("[INFO] loading the pre-trained model")
#model = load_model(args["model"])

#classification reports on training data

y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = model.predict(test_dataset, batch_size=64)

y_pred[y_pred >= 0.5] = 1  # converting all the positive cases
y_pred[y_pred < 0.5] = 0  # converting all the negative cases

y_true = list(y_true.ravel())
y_pred = list(y_pred.ravel())

print(classification_report(y_true, y_pred, 
    target_names=['normal', 'pneumonia']))

print('Confusion matrix')
print(confusion_matrix(y_true, y_pred))

print("[INFO] saving the model...")
model.save(args["model"])

# finding predictions for normal test images
print('Correctly classified images(normal): ', predict_test_set('xray_dataset/test/normal/', normal=True))
print('Correctly classified images(pneumonia): ', predict_test_set('xray_dataset/test/pneumonia/', normal=False))

# plotting the evaluation results
epochs_range = range(epochs)
plt.figure()
plt.plot(epochs_range, H.history['accuracy'], label='Training accuracy')
plt.plot(epochs_range, H.history['val_accuracy'], label='Validation accuracy')
plt.plot(epochs_range, H.history['loss'], label='Training loss')
plt.plot(epochs_range, H.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss/accuracy')
plt.legend()
plt.savefig(args["output"])
