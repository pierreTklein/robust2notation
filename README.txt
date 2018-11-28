DIRECTIONS:
-Have installed Python 3 to 3.6
-If dependencies are not met, run `./setup.sh`.

PREPROCESSING:
-Run preprocessing.ipynb in Jupyter or preferred interface to preprocess training data.

NUMPY NEURAL NET:
This is a fully connected, feed-forward neural net, implemented from scratch using only Numpy.
-If not already done run preprocessing.ipynb.
-Run handmade_NN.ipynb in Jupyter or preferred interface to train the NN.

KAGGLE SUBMISSION CNN:
-If not already done, run preprocessing.ipynb.
-Run `py concurrent_rotate.py` to generate rotations/scalings of training data.
-Run Jupyter notebook and open cnn-final.ipynb.
-You can run the code from the top down, though look at comments for
	instructions specific to running a new/saved model generated
	by the notebook code.
	
FILES:
./
Classical_Models.ipynb : Python notebook for the classical methods used including
						linear SVM, random forest, and k-nearest neighbors.
cnn-final.ipynb : Final Python notebook showcasing our best-predicting model on Kaggle.
cnn-working.ipynb : A non-refactored notebook with our raw work for older models and the submitted one.
CNN.ipynb : A non-refactored notebook with our raw work for older models.
concurrent_rotate.py : Non-representative name. Takes provided training data and generates,
						for each image, 5 scalings of it including the default, and 12
						rotations of 30 degree intervals including 0.
						Creates 600,000 images out of 10,000 provided.
						Creates multiple processes.
conv_autoencoder.ipynb : contains a convolutional autoencoder, which was tested as a potential de-noising option.
data.zip : Contains original training and unlabeled data from the competition. Used by preprocessing.ipynb.
handmade_NN.ipynb : contains the Jupyter Notebok used in the training.
homecooked_NN.py : contains the Neural net class used by handmade_NN.ipynb.
preprocessing.ipynb : Preprocesses raw provided image data into .npy files in /data.
