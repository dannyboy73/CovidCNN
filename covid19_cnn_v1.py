from tensorflow.keras.preprocessing import image_dataset_from_directory # for importing images form the directory
from tensorflow.keras.models import Sequential # used to group a stack of layers
from tensorflow import expand_dims # to expand the shape of image array
from keras.preprocessing import image # functions for image pre-processing
from tensorflow.keras.models import load_model # to import the necessary model
from tensorflow.keras.optimizers import Adam # optimization algorithm for the model
from sklearn.metrics import confusion_matrix # metrics to evaluate the model
import argparse
import os
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

def classification_report(path):
	'''Used for predicting images in the test set and find how many are classified correctly w.r.t to each class'''
	img = []
	y_true = np.zeros(100, dtype=int)
	y_true = np.concatenate((y_true, np.ones(100, dtype=int)), axis=0)
	path_format = path + '{0}'
	ext = ['normal/', 'pneumonia/']
	for i in ext:
		for filepath in os.listdir(path+i):
			x = image.load_img(path+i+filepath, target_size=(256, 256),
					    color_mode="grayscale")
			x = image.img_to_array(x)
			x = expand_dims(x, 0)
			img.append(x)
		
	preds = []
	for x, i in enumerate(img):
		predictions = model.predict(i)
		preds.append(predictions[0][0])
		print(predictions, y_true[x])
		if preds[x] >= 0.5: preds[x] = 1
		else: preds[x] = 0
	
	tot_images = 200
	tp, tn, fn, fp = 0, 0, 0, 0
	for i in range(200):
		if preds[i] == 1 and y_true[i] == 1:
		    tp += 1
		if preds[i] == 0 and y_true[i] == 0:
		    tn += 1
		if preds[i] == 1 and y_true[i] == 0:
		    fn += 1
		if preds[i] == 0 and y_true[i] == 1:
		    fp += 1
		    
	print("-------------------------------")
	print("|Accuracy: ", str((tp+tn)/tot_images))
	print("|Precision: ", str(tp/(tp+fp)))
	print("|Recall: ", str(tp/(tp+fn)))
	print("|F1: ", str(2*(tp/(tp+fp)*tp/(tp+fn))/(tp/(tp+fp)+tp/(tp+fn))))
	print("-------------------------------")
	

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
        help="path to the input dataset")
ap.add_argument("-m", "--model", required=True,
        help="path to load/save the trained model")
args = vars(ap.parse_args())

img = []
path = []
path.append(args["dataset"] + 'train/normal/')
path.append(args["dataset"] + 'train/pneumonia/')

# loading the train image data (80% for train, 20% for validation)
train_dataset = image_dataset_from_directory(
        args["dataset"], labels='inferred', label_mode='binary',
        color_mode="grayscale", image_size=(256,256), subset="training",
        seed=7, validation_split=0.2)

# loading the validation image data 
validation_dataset = image_dataset_from_directory(
        args["dataset"], labels='inferred', label_mode="binary",
        color_mode="grayscale", image_size=(256,256), subset="validation",
        seed=7, validation_split=0.2)

# loading the test image data
test_dataset = image_dataset_from_directory(
        "xray_dataset/test", labels='inferred', label_mode='binary',
        color_mode="grayscale", image_size=(256,256), seed=7)

class_names = train_dataset.class_names # obtain the different classes (Normal, Pneumonia)
print(class_names)

'''
# Image pre-processing technique used
# 1. Data augmentation
'''
data_augmentation = Sequential([
    #layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    #layers.RandomContrast(0.1),
])

# Note: The other data augmentation techniques haven't helped increase the performance of the model

# building the model. we apply standardization within the model
model = Sequential([
    # applying the data augmentation layer
    data_augmentation,
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

'''
# compiling the model
#opt = SGD(lr=0.05, decay=0.05, momentum=0.9, nesterov=True)
opt = Adam(learning_rate=0.0005)
model.compile(optimizer=opt,
        loss='binary_crossentropy',
        metrics=["accuracy"])


# training the model
print("[INFO] training the network...")
epochs = 10
H = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        batch_size=64,
        epochs=epochs,
        verbose=1,
    )
'''

model = load_model(args["model"])

model.summary()


#print("[INFO] saving the model...")
#model.save(args["model"])

#classification reports on training data

'''
y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = model.predict(test_dataset, batch_size=64)

y_pred[y_pred >= 0.6] = 1  # converting all the positive cases
y_pred[y_pred < 0.6] = 0  # converting all the negative cases

y_true = list(y_true.ravel())
y_pred = list(y_pred.ravel())

print(y_true)
print(y_pred)
'''

# create a function for classification_report and confusion matrix

print('Classification report')
classification_report('xray_dataset/test/')

print('Confusion matrix')
#print(confusion_matrix(y_true, y_pred))

'''
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1): 
    print(images[0])
    for i in range(9):
        image = tf.expand_dims(images[0], 0)
        print(image)
        augmented_image = data_augmentation(image)
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(augmented_image[0].numpy().astype("uint8"), cmap='gray')
        plt.axis("off")

plt.show()
'''

