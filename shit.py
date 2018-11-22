from scipy.ndimage import zoom

# Finds the first non-zero coordinate, and the last non-zero coordinate.


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
    print(zoomFactor)
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


shovel = np.load('./data/shovel.npy')
shovel = (shovel * 255).astype(int)
sharpenedImg = sharpen(shovel, 0)
cropped = cropWhite(sharpenedImg, True)
centered = center(cropped)
rescaled = rescale(centered, 80, 1)
plt.imshow(rescaled, cmap='gray_r')
plt.show()
