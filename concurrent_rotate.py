from math import floor, ceil
from keras.utils import to_categorical
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import concurrent.futures

from scipy.ndimage import zoom, rotate

TRAIN_LAB_PATH = "./data/train_labels.csv"
PREPROCESSED_TRAINING = "./data/processedData.npy"
PREPROCESSED_KAGGLE = "./data/processed_kaggle.npy"
ROTATED_LAB = "./rotatedTrainLabels.csv"
ROTATED_TRAINING = "./rotatedTrainData.npy"

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
	return (minX, minY), (maxX, maxY)

# Centers the image


def center(img):
	minCoord, maxCoord = boundingBox(img)
	xLength = maxCoord[0] - minCoord[0]
	yLength = maxCoord[1] - minCoord[1]
	newImg = [[0 for j in range(len(img[i]))] for i in range(len(img))]

	startX = int((len(img) - xLength) / 2)
	startY = int((len(img[0]) - yLength) / 2)
	for i, x in enumerate(range(startX, startX + xLength + 1)):
		for j, y in enumerate(range(startY, startY + yLength + 1)):
			newImg[x][y] = img[minCoord[0]+i][minCoord[1]+j]
	return newImg

# Crop out all of the white space. If you want square dimensions, then it will pad white space.


def cropWhite(img, isSquare=False, whiteBoundary=True):
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


def rescale(img, dimension, order=0):
	cropped = cropWhite(img)
	height = len(cropped)
	width = len(cropped[0])
	zoomFactor = dimension / max(height, width)
	# print(zoomFactor)
	return zoom(img, zoomFactor, order=order)


def sharpen(img, cutoff=110, bound='below'):
	newImg = []
	for row in img:
		newRow = []
		for pixel in row:
			if pixel < cutoff and (bound == 'below' or bound == 'both'):
				newRow.append(0)
			elif pixel < cutoff:
				newRow.append(pixel)
			elif pixel > cutoff and (bound == 'above' or bound == 'both'):
				newRow.append(255)
			else:
				newRow.append(pixel)

		newImg.append(newRow)
	return np.asarray(newImg)

def getRotations(x, y, rescaleDimension = 40, order = 0, interval_deg=30):
	newX = []
	newY = []
	rescaled_img = (x * 255).astype(int)
	newX.append(rescaled_img.reshape((rescaleDimension, rescaleDimension)))
	newY.append(y)
	deg = interval_deg
	while deg < 360:
		# Rotate and fix
		rotImg = rotate(x, deg)
		img255 = (rotImg * 255).astype(int)
		croppedImg = cropWhite(img255, True)
		centered_img = center(croppedImg)
		rescaled_img = rescale(centered_img, rescaleDimension, order)
		rescaled_img[rescaled_img < 0] = 0
		rescaled_img[rescaled_img > 255] = 255
		newX.append(rescaled_img)
		newY.append(y)
		deg += interval_deg
	return newX, newY

def getScalings(x, y, rescaleDimension=40, dims=(34,30,24,18), order=0):
	newX = []
	newY = []
	newX.append(x)
	newY.append(y)
	pads = {34,30,24,18}
	for dim in dims:
		# img255 = (x * 255).astype(int)
		# sharpenedImg = sharpen(img255, 0)
		rescaled_img = rescale(x, dim, 1)
		croppedImg = cropWhite(rescaled_img, True)
		rescaled_img = rescale(croppedImg, rescaleDimension, order)
		# centered_img = center(croppedImg)

		# pad_img = np.pad(rescaled_img, pad_width=pads[dim], mode='constant')

		# print(pad_img.shape, dim)
		# rescaled_img = rescale(pad_img, rescaleDimension, order)

		newX.append(rescaled_img)
		newY.append(y)

	return newX, newY


def getExtendedData(X, Y, rescaleDimension = 40, interval_deg=30):
	with concurrent.futures.ProcessPoolExecutor() as executor:
		newX = []
		newY = []
		for newx, newy in executor.map(getRotations, X, Y):
			newX.extend(newx)
			newY.extend(newy)

		newX = np.array(newX)

		resultX = []
		resultY = []
		for newx, newy in executor.map(getScalings, newX, newY):
			resultX.extend(newx)
			resultY.extend(newy)

	return resultX, resultY

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
	X, y = formatData(training_imgs, labels)
	Xnew, ynew = getExtendedData(X,
								 y)


	save("extendedTrainData.npy", X)
	np.savetxt("extendedTrainLabels.csv", y)
