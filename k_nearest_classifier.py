import common_methods_lib as cml

import random
import math
import time
import operator

def getPercLabels(labelList, randRange):
    labtrix = []
    for n in randRange:
        labtrix.append(labelList[n])

    return labtrix

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

def classify(neighbors, trainSet, trainLabelList):
    hashVotes = {}
    for x in range(len(neighbors)):
        vote = trainLabelList[trainSet.index(neighbors[x])]
        
        if vote in hashVotes:
            hashVotes[vote] += 1
        else:
            hashVotes[vote] = 1

    #now sort votes to figure out most favored
    #votes = sorted(hashVotes.items(), key = operator.itemgetter(1))
    #return votes[0][0]
    return max(hashVotes, key=hashVotes.get)

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
labelList = cml.readInLabels(trainLabelsfn)

#testing data, labels, and feature vectors
testingMatrix = cml.readInFile(testfn, testCount, testHeight)
testLabels = cml.readInLabels(testLabelsfn)

testFeatVects = cml.getFeatureVectors(testingMatrix, cml.pixelsPerLine)

#start the training data at 10%
percent = 0.1

while percent <= 1:
    currRange = int(round(percent, 1) * len(labelList))

    count = 0
    average = 0.0
    corrects = []
    
    while(count < 5):
        #get a list of numbers of currRange length which can range from 0 to number of labels/images
        randSamp = random.sample(range(len(labelList)), currRange)
        
        #get a whatever% sample of the label list and traintrix, importantly same indeces
        trainingLabels = getPercLabels(labelList, randSamp)
        trainingImages = cml.randTrainImgs(trainMatrix, randSamp)
        
        #get the feature vectors for each image
        trainFeatVect = cml.getFeatureVectors(trainingImages, cml.pixelsPerLine)

        #Run for every element in test
        numCorrect = 0
        for x in range(len(testLabels)):
            neighbors = getNeighbors(trainFeatVect, testFeatVects[x], 3)
            guess = classify(neighbors, trainFeatVect, trainingLabels)

            if guess == testLabels[x]:
                #print(f'Correct\n')
                numCorrect += 1
                        
        average += numCorrect
        count += 1
        corrects.append(numCorrect)

    average = round(average/count, 0)
    variance = cml.calcVariance(corrects, average)

    print(f'Statistics for {round(100*percent, 1)}% training data with {count} trials')
    print(f'\tAverage correct guesses: {average} out of {len(testLabels)} correctly')
    print(f'\tVariance: {variance}')
    print(f'\tStandard Deviation: {math.sqrt(variance)}\n')

    percent += 0.1


print(f'This took {round(time.time() - start_time, 2)} seconds')