# vehicle-cnn
Vehicle Classification CNN using Keras. This is my first actual implemented network where I was still experimenting and learning the ropes, while trying to grasp the basic of neural networks while I was working. I've tried experimenting with defining a model, fine tuning, and also self-implementing a pre-trained network, while also trying to gather various methods for data loading and model prediction. Some of the functions implemented here are still useful for me till this very day, so I decided to keep this as a repository. 
There is nothing confidential here, so I decided to make it public after some code cleaning up. Sorry for the storytelling, I don't think it's common to do so in Git.

### 1) Preprocess Data and Extract Data
* `prepare_data.py` - Load data from data into arrays of label and features, then creating training and testing datasets. The training and testing datasets are saved into NumPy arrays.

### 2) Build Model and Run Training
* `alexnet.py` - Self-implemented AlexNet model building and training.
* `finetune.py` - Fine tuning with Keras' prebuilt VGG-19 model, then adding convolution layers after the VGG-19 model output. The dataset are loaded using Keras' flow_from_directory method, instead of the one as in `prepare_data.py`.
* `model.py` - Self defined shallow model for training. A basic function to plot the training and testing accuracy and loss is defined here.

### 3) Making Prediction
* `prediction.py` - Making predictions using saved model and calculating the accuracy. There are 2 methods to perform the prediction: predicting the test image array and then comparing it with the test labels, and loading the images from a directory and compare the prediction of the images with the image directory as ground truth.

### \*) Miscellaneous
* `tester.py` - Testing if Tensorflow is running on GPU.
