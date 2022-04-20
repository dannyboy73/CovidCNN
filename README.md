# CovidCNN
Binary classification to classify whether the given X-ray can be diagnosed as Pneumonia (Covid induced) or not.

As a beginner in the field of deep learning, I'd decided to delve into the subject matter with a practical approach and that is how I started with this project. A book that was of great help during the project was Adrian Rosebrock's Deep Learning for Computer Vision (link at the end). Finding a dataset online was a task as most of the links were broken or didn't have sufficient images (<1000). So, I collected all the images from different sources and collated them. I'm not a radiologist and have no idea how to diagnose the X-ray's but I felt that the classifier could at least do a much better job than me. As for the Neural Network, the entire architecture can be summarized as:

INPUT => (CONV => RELU => MAXPOOLING) * 3 => DROPOUT => (CONV => RELU => MAXPOOLING) * 2 => DROPOUT => FC => DROPOUT => SIGMOID

### Detailed model architecture

![alt text](https://github.com/dannyboy73/CovidCNN/blob/main/images/model_architecture.png)

The reason for dropouts in between the first convolutional layer, second convolutional layer and the fully connected layer is put in place to reduce overfitting by the model on the training images.
After messing with different optimizers (SGD, Custom learning rate schedulers, Adam, RMSProp), Adam optimizer (with a learning rate of 0.0001) performs really well compared to the rest .

### Training loss and accuracy plot

![alt text](https://github.com/dannyboy73/CovidCNN/blob/main/output/covid.png)

From the evaluation plot it's evident that the training accuracy of the model falls below 0.2 at the end of 15 epochs. We stop the model right beofre the validation loss starts to form a gap with respect to training loss. The training accuracy and validation accuracy form a consistent linear trend after 4 epochs. 

### Classification report

![alt text](https://github.com/dannyboy73/CovidCNN/blob/941735f523758c729dc8d5a20b498fb59740b3ba/images/classification_report.png)

Looking at the classification report, precision = 0.98, recall = 0.88, accuracy = 0.925, and F1 score = 0.928.

**Web APP coming soon!**
