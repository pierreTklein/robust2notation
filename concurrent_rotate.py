import sys
from keras.utils import to_categorical
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import concurrent.futures

TRAIN_LAB_PATH = "./data/train_labels.csv"
PREPROCESSED_TRAINING = "./data/processedData.npy"
PREPROCESSED_KAGGLE = "./data/processed_kaggle.npy"

CATEGORIES = ['apple', 'empty', 'moustache', 'mouth', 'mug', 'nail', 'nose', 'octagon', 'paintbrush', 'panda', 'parrot', 'peanut', 'pear', 'pencil', 'penguin', 'pillow', 'pineapple', 'pool', 'rabbit', 'rhinoceros', 'rifle', 'rollerskates', 'sailboat', 'scorpion', 'screwdriver', 'shovel', 'sink', 'skateboard', 'skull', 'spoon', 'squiggle']

def getIndexOf(category):
	return CATEGORIES.index(category)

def getCategoryOf(index):
	return CATEGORIES[index]

def load(infile):
	unformatted_images = np.load(infile, encoding='bytes')
	formatted = []
	for i,img in enumerate(unformatted_images):
		formatted.append([i, img[0]])
	return formatted

def save(outfile, images):
	reformattedImgs = []
	for i, image in enumerate(images):
		reformattedImg = image.reshape((1,image.shape[0] * image.shape[1]))
		reformattedImgs.append(reformattedImg)
	nparr = np.asarray(reformattedImgs)
	np.save(outfile, nparr)

def formatXData(X, xDimension = 40):
	X = np.asarray(X)
	# Convert to matrix form
	X = X.reshape(-1, xDimension, xDimension, 1)
	# Convert to float
	X = X.astype('float32')
	# Scale pixel values between 0 and 1
	X = X / 255
	return X.astype('float32')

def addRotations(X,y):
	newX = []
	newY = []
	for i,XMatrix in enumerate(X):
		newX.append(XMatrix)
		newY.append(y[i])
		newX.append(np.rot90(XMatrix, 1))
		newY.append(y[i])
		newX.append(np.rot90(XMatrix, 2))
		newY.append(y[i])
		newX.append(np.rot90(XMatrix, 3))
		newY.append(y[i])
	return np.asarray(newX),np.asarray(newY)

from scipy.ndimage import rotate, zoom

#Finds the first non-zero coordinate, and the last non-zero coordinate.
def boundingBox(img):
	minX = -1
	minY = -1
	maxX = -1
	maxY = -1
	for i, row in enumerate(img):
		nonZeroIndexes = np.nonzero(row)[0]
		if len(nonZeroIndexes) != 0:
			if minX == -1:
				minX = i
			if minY == -1 or minY > np.min(nonZeroIndexes):
				minY = np.min(nonZeroIndexes)
			if maxX < i:
				maxX = i
			if maxY < np.max(nonZeroIndexes):
				maxY = np.max(nonZeroIndexes)
	return (minX, minY), (maxX,maxY)

# Centers the image
def center(img):
	minCoord, maxCoord = boundingBox(img)
	xLength = maxCoord[0] - minCoord[0]
	yLength = maxCoord[1] - minCoord[1]
	newImg = [[0 for j in range(len(img[i]))] for i in range(len(img))]

	startX = int((len(img) - xLength) / 2)
	startY = int((len(img[0]) - yLength) / 2)
	for i,x in enumerate(range(startX, startX + xLength + 1)):
		for j,y in enumerate(range(startY, startY + yLength + 1)):
			newImg[x][y] = img[minCoord[0]+i][minCoord[1]+j]
	return newImg

# Crop out all of the white space. If you want square dimensions, then it will pad white space.
def cropWhite(img, isSquare = False, whiteBoundary = True):
	minCoord, maxCoord = boundingBox(img)
	xLength = maxCoord[0] - minCoord[0] + 3
	yLength = maxCoord[1] - minCoord[1] + 3
	if isSquare:
		xLength = max(xLength, yLength)
		yLength = max(xLength, yLength)

	newImg = [[0 for j in range(yLength + 1)] for i in range(xLength + 1)]
	for i in range(xLength):
		for j in range(yLength):
			# Check for case where we are out of bounds for cropped white + square
			if (minCoord[0] + i) >= len(img) or (minCoord[1] + j) >= len(img[i]):
				newImg[i + 1][j + 1] = 0
			else:
				newImg[i + 1][j + 1] = img[minCoord[0] + i][minCoord[1] + j]
	return newImg

# rescale image to square of height, width = dimension
def rescale(img, dimension, order = 0):
	cropped = cropWhite(img)
	height = len(cropped)
	width = len(cropped[0])
	zoomFactor = dimension / max(height, width)
	return zoom(img, zoomFactor, order=order)

def getRotations(x, y, rescaleDimension = 40, order = 1, interval_deg=30):
	newX = []
	newY = []
	newX.append(x)
	newY.append(y)
	deg = interval_deg
	while deg < 360:
#         newX.append(np.rot90(XMatrix, 1))
		rotImg = rotate(x, deg)
		croppedImg = cropWhite(rotImg, True)
		centered_img = center(croppedImg)
		rescaled_img = rescale(centered_img, rescaleDimension, order)
		newX.append(rescaled_img)
		newY.append(y)
		deg += interval_deg
	return newX, newY

def getRotData(X, Y, rescaleDimension = 40, interval_deg=30, parallel=True):
	newX = []
	newY = []
	if parallel:
		with concurrent.futures.ProcessPoolExecutor() as executor:
			for newx, newy in executor.map(getRotations, X, Y):
				newX.extend(newx)
				newY.extend(newy)
	else:
		for i in range(0,len(X)):
			newx, newy = getRotations(X[i], Y[i], rescaleDimension=rescaleDimension, interval_deg=interval_deg)
			newX.extend(newx)
			newY.extend(newy)
	return newX, newY

def formatData(images, labels, xDimension = 40):
	categories = list(set(labels['Category']))
	X = []
	y = []
	for i, img in enumerate(images):
		label = labels.at[i,'Category']
		categoryNum = getIndexOf(label)
		X.append(img[1])
		y.append(categoryNum)
	y = to_categorical(y)
	X = formatXData(X, xDimension)
	return X.astype('float32'), y

if __name__ == '__main__':
	training_imgs = load(PREPROCESSED_TRAINING)
	labels = pd.read_csv(TRAIN_LAB_PATH)
	X,y = formatData(training_imgs, labels)
	X,y = getRotData(X[0:10], y[0:10])
	save("rotatedTrainData.npy", X)
	np.savetxt("rotatedTrainLabels.csv", y)
