import common_methods_lib as cml

import random
import math
import time
import operator

def euclidDistance(v1, v2):
    distance = 0

    for x in range(len(v1)):
        distance += pow(v1[x] - v2[x], 2)
    return math.sqrt(distance)

def getNeighbors(trainSet, test, n):
    distances = []
    
    for x in range(len(trainSet)):
        dist = euclidDistance(trainSet[x], test)
        distances.append((trainSet[x], dist))
    #order by proximity to test
    distances.sort(key=operator.itemgetter(1))
    
    #return vector, load nearest n neighbors
    nearestNeighbors = []
    for x in range(n):
        nearestNeighbors.append(distances[x][0])
    
    return nearestNeighbors

def classify(neighbors, trainSet, trainLablelList):
    hashVotes = {}
    for x in range(len(neighbors)):
        vote = trainLabelList[trainSet.index(neighbors[x])]
        
        if vote in hashVotes:
            hashVotes[vote] += 1
        else:
            hashVotes[vote] = 1

    #now sort votes to figure out most favored
    votes = sorted(hashVotes.items(), key = operator.itemgetter(1), reverse = True)
    return votes[0][0]

#MAIN METHOD
start_time = time.time()

#get user's input
which = input('\"face\" or \"digit\"?\n')
while which != 'face' and which != 'digit':
    which = input('Invalid input. Please input \"face\" or \"digit\" (no quotes)\n')

#preliminary definitions
trainfn = ''
trainLabelsfn = ''
trainCount = 0
trainHeight = 0
testfn = ''
testLabels = ''
testCount = 0
testHeight = 0

if which == 'face':
    trainfn = 'facedata/facedatatrain'
    trainLabelsfn = 'facedata/facedatatrainlabels'
    trainCount = 451
    trainHeight = 70
    testfn = 'facedata/facedatatest'
    testLabelsfn = 'facedata/facedatatestlabels'
    testCount = 150
    testHeight = 70

elif which == 'digit':
    trainfn = 'digitdata/trainingimages'
    trainLabelsfn = 'digitdata/traininglabels'
    trainCount = 5000
    trainHeight = 28
    testfn = 'digitdata/testimages'
    testLabelsfn = 'digitdata/testlabels'
    testCount = 1000
    testHeight = 28

else:
    print('invalid input')
    exit(0)

#training data and labels
trainMatrix = cml.readInFile(trainfn, trainCount, trainHeight)
trainLabelList = cml.readInLabels(trainLabelsfn)

#testing data, labels, and feature vectors
testingMatrix = cml.readInFile(testfn, testCount, testHeight)
testLabels = cml.readInLabels(testLabelsfn)

testFeatVects = cml.getFeatureVectors(testingMatrix, cml.pixelsPerLine)

#start the training data at 10%
percent = 1

while percent <= 1:
    currRange = int(round(percent, 1) * len(trainLabelList))

    #get a list of numbers of currRange length which can range from 0 to number of labels/images
    randSamp = random.sample(range(len(trainLabelList)), currRange)
    
    #get a whatever% sample of the label list and traintrix, importantly same indeces
    #percOfLabList = cml.getNumCounts(trainLabelList, randSamp)
    trainingImages = cml.randTrainImgs(trainMatrix, randSamp)
    
    #get the feature vectors for each image
    trainFeatVect = cml.getFeatureVectors(trainingImages, cml.pixelsPerLine)

    #Run for every element in test
    numCorrect = 0
    for x in range(len(testLabels)):
        neighbors = getNeighbors(trainFeatVect, testFeatVects[x], 3)
        guess = classify(neighbors, trainFeatVect, trainLabelList)

        if guess == testLabels[x]:
            print(f'Correct\n')
            numCorrect += 1
        else:
            print(f'Incorrect\n')

    print(f'Statistics for {round(100*percent, 1)}%')
    print(f'\tCorrect guesses: {numCorrect} out of {len(testLabels)}')
    percent += .1


print(f'This took {round(time.time() - start_time, 2)} seconds')