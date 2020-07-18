# vehicle-cnn
Vehicle Classification CNN using mainly Tensorflow

### 1) Preprocess Data and Extract Data
* `extractfeatures.py` - Load data into arrays of label and features, then creating training and testing datasets

### 3) Build Model and Run Training
* `alexnet.py` - Self-implemented AlexNet model building and training.
* `finetune.py` - Self-implemented VGG19 model building and training, with transfer learning.
* `model.py` - Self defined shallow model for training.

### 4) Making Prediction
* `prediction.py` - Making predictions using saved model.

### \*) Miscellaneous
* `tester.py` - Testing if Tensorflow is running on GPU.
