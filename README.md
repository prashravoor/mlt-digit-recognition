# Handwritten Digit Recognition (MNIST DB) using CNN and SVM

## Configuration
Python version 3.6+

## Setup
Install all the requirements using `pip install -r requirements.txt`.

## Running the code
You can start the GUI using `python gui.py`. The GUI has configurable parameters to set:
* Model to use - CNN or SVM
* Train Data set size - 100 - 60000
* Test Data set size - 50 - 10000

There are also actions to train the model (model is automatically saved to a file), predict a single image, or generate a confusion matrix using the value specified in the `Test Size` field. <br>
In addition, "Predict Batch" allows selection of multiple images in one go, and displays results for all of them on the screen. <br>
"Predict Live" will open the webcam. The digit can be shown to the screen, and press `s` to capture the image. This image will be classified and the result is diplayed on the screen. <br>

Alternatively, you can run the CLI using the corresponding python files for CNN or SVM <br>
* `python cnn.py {train | predict | confusion} {filename to load/save} {Train Dataset Size} {Test Dataset Size} {Number of Epochs}`
* `python svm.py {train | predict | confusion} {filename to load/save} {Train Dataset Size} {Test Dataset Size}`

All parameters, except the first one (train/predict/confusion) are optional. Suitable defaults are used if they are not specified. Some examples of usage are shown below. <br>

* Train CNN Model, with train size = 10000, test size = 1000, Number of epochs = 10 <br>
  `python cnn.py train cnn.hdf5 10000 1000 10`
* Train SVM Model, with train size = 10000, test size = 1000<br>
  `python svm.py train svm.dump 10000 1000`
* Train SVM Model, with train size = 10000, test size = 1000, and search for correct parameters<br>
  `python svm.py train svm.dump 10000 1000 search`
* Predict single image Using SVM model <br>
  `python svm.py predict <path to image file> svm.dump`
* Show confusion Matrix for CNN model using 1000 images as test size <br>
   `python cnn.py confusion 1000`
  
## Generating Performance Graphs
Running `python eval_perf_cnn.py` and `python eval_perf_svm.py` generate performance and parameter evaluation graphs for CNN and SVM models respectively <br>
Both could take a LOT of time to complete (Up to 6 hours for CNN). CNN script will additionally produce graphs that show how varying parameters affect accuracy. <br>