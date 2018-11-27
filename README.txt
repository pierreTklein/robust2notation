DIRECTIONS:
-Have installed Python 3 to 3.6
-If dependencies are not met, run `./setup.sh`.

PREPROCESSING:
-Run preprocessing.ipynb in Jupyter or preferred interface to preprocess  training data.

<IMPLEMENTED NN / BASELINES>:
-insert directions here

KAGGLE SUBMISSION CNN:
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
cnn-working.ipynb : A non-refactored notebook with our raw work for the submitted model.
concurrent_rotate.py : Non-representative name. Takes provided training data and generates,
						for each image, 5 scalings of it including the default, and 12
						rotations of 30 degree intervals including 0.
						Creates 600,000 images out of 10,000 provided.
						Creates multiple processes.
conv_autoencoder.ipynb : INSERT DESCRIPTION HERE
handmade_NN.ipynb : Code for the hand-written neural network.
preprocessing.ipynb : Preprocesses raw provided image data into .npy files in /data.

data/
data.zip : Contains original training and unlabeled data from the competition. Used by preprocessing.ipynb.

829model/
replica_pred.csv : Predictions from a replication of our 82.9-scoring CNN. Not used by anything.
model5-40ep-10k-Replica.h5 : Model file of our highest-scoring CNN. Not used by anything.