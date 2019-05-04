import numpy as np

#returns 3D matrix containing all training data
def readInFile(filename, dataCount, dataHeight):
    digitsMatrix = []
    digitsInfo = open(filename, 'r')
    for i in range(dataCount):
        digitsMatrix.append([])
        for j in range(dataHeight):
            line = digitsInfo.readline()
            #removes final space for ease of manipulation later
            line = line[:-1]
            #trims out line of only whitespace, delete later if bad/test how results change
            #if(line.isspace()): continue
            digitsMatrix[i].append(list(line))
    digitsInfo.close()

    return digitsMatrix

#read in training labels, return as list
def readInLabels(filename):
    labelsList = []
    labels = open(filename, 'r')
    for label in labels:
        labelsList.append(int(label))

    return labelsList

#gets count of each number in the training set according to our random range
def getNumCounts(labelsList, randRange):
    numCounts = {}

    for n in randRange:
        num = int(labelsList[n])
        if num in numCounts:
            numCounts[num] += 1
        else:
            numCounts[num] = 1
    return numCounts

#picks a percentage of the training images in random fashion
def randTrainImgs(traintrix, randRange):
    percTrix = []
    for n in randRange:
        percTrix.append(traintrix[n])

    return percTrix

#feature - calculate pixel density / or just count number of black pixels
def densityFeatures(imageLines):
    blackPixels = 0
    whitePixels = 0
    for line in imageLines:
        for char in line:
            if char != ' ':
                blackPixels += 1
            else:
                whitePixels += 1

    return (blackPixels, whitePixels)

#feature - one feature for each line in image, counts number of marked/unmarked pixels
def pixelsPerLine(imageLines):
    feats = []
    for line in imageLines:
        blackPix = 0
        for char in line:
            if char != ' ':
                blackPix += 1
        feats.append(blackPix)
    return feats

#features - partition digit image into 4x7 grid and return binary list based on
#if the particular square has any spots in it or not
#this only works for digits sorry
def partitionFeatures(imageLines):
    partitionFeatures = [0]

    iArray = np.asarray(imageLines)
    gridImage = blockshaped(iArray, 7, 4)

    for i in range(len(gridImage)):
        for j in range(len(gridImage[i])):
            if ('+' or '#') in gridImage[i][j]:
                partitionFeatures[i] += 1
        partitionFeatures.append(0)

    return partitionFeatures

#utility function to partition array into grids for partitionGrid method
def blockshaped(arr, nrows, ncols):
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

#feature that I might use, one for each pixel
def featurePerPixel(imageLines):
    pixelFeatures = []
    for line in range(len(imageLines)):
        for char in range(len(imageLines[line])):
            if imageLines[line][char] != ' ':
                pixelFeatures.append(1)
            else:
                pixelFeatures.append(0)
    return pixelFeatures

#calculate feature vector for each training data entry based on function passed in
def getFeatureVectors(datatrix, featureFun):
    features = []
    for entry in datatrix:
        featureVect = featureFun(entry)
        features.append(featureVect)
    return features

#count times a feature occurs for each grid for each label in training images of random
def countFeatures(featureList, labelsList, randSamp):
    fDict = {}

    for i in range(len(featureList)):
        currFeatVect = featureList[i]
        currLabel = labelsList[randSamp[i]]
        for j in range(len(currFeatVect)):
            want = currFeatVect[j]
            if want not in fDict:
                fDict[want] = {}
            if j not in fDict[want]:
                fDict[want][j] = {}
            if currLabel not in fDict[want][j]:
                fDict[want][j][currLabel] = 1
            else:
                fDict[want][j][currLabel] += 1
    return fDict 

def calcVariance(corrects, average):
    variance = 0
    for correct in corrects:
        variance += (correct - average)**2
    variance /= len(corrects)
    
    return variance