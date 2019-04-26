import common_methods_lib as cml

import numpy as np
import random
import math
import time
            
#test input is binary features of one test image
#featureCount is info gathered from training
#numCount to divide for naive bayes calculations
def naiveBayes(testInput, featureDict, numCounts):

    partitionPossibilities = []

    for i in range(len(testInput)):
        partitionPossibilities.append(allForOne(testInput[i], i, featureDict, numCounts))

    guessPossibilities = []
    for i in range(len(numCounts)):
        pos = 1
        for j in range(len(partitionPossibilities)):
            pos *= partitionPossibilities[j][i]
        guessPossibilities.append(pos)

    guess = guessPossibilities.index(max(guessPossibilities))
    return guess

#get range of possibilities over values 0-9 or 0-1 (digits/face) for nth partition
#refer to slide 32/40 for intro to ML if this needs clarification
def allForOne(binary, partition, featureDict, numCounts):
    tot = sum(numCounts)
    partitionPossibs = {}
    for i in range(len(numCounts)):
        numer = 0
        if binary in featureDict and partition in featureDict[binary] and i in featureDict[binary][partition]:
            numer = featureDict[binary][partition][i]
        denom = numCounts[i]
        #this probability is feature(x) = nth partition given either 0 or 
        # 1 divided by number of times x appears in teh training data set
        # we will apply laplace smoothing so as to not get a prob of ZERO 
        partitionPossibs[i] = (numer + 1)/(denom + tot)
    return partitionPossibs

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

print("SO BEGINS THE TRAINING")

#start the training data at 10%
percent = 0.1

while percent <= 1:
    currRange = int(round(percent, 1) * len(labelList))

    #need to average our results for each percent of training we take in, lets try 10 for now
    count = 0
    average = 0.0
    #keep track of # of times classifier guessed correctly for each trial to calculate variance and then sd
    corrects = []

    while(count < 10):
        #get a list of numbers of currRange length which can range from 0 to number of labels/images
        randSamp = random.sample(range(len(labelList)), currRange)
        
        #get a whatever% sample of the label list and traintrix, importantly same indeces
        percOfLabList = cml.getNumCounts(labelList, randSamp)
        percOfTraintrix = cml.randTrainImgs(trainMatrix, randSamp)

        #get the feature vectors for each image
        trainFeatVect = cml.getFeatureVectors(percOfTraintrix, cml.pixelsPerLine)

        #dictionary has all data necessary for calculating naive bayes
        #ex: bDict[0][0][0]: tells you how many images labelled 0 have no pixels marked in the 0th grid partition
        #ex: bDict[1][4][5]: tells you how many images labelled 5 have at least 1 marked pixel in the 4th grid partition
        bDict = cml.countFeatures(trainFeatVect, labelList, randSamp)

        """COMMENCE GUESSAGE"""
        correctCount = 0
        for i in range(len(testFeatVects)):
            guess = naiveBayes(testFeatVects[i], bDict, percOfLabList)
            if guess == testLabels[i]:
                correctCount += 1
        #add to the average
        average += correctCount
        count += 1
        corrects.append(correctCount)
        
    #divide average by count to get average over the ten trials
    average = round(average/count, 0)
    variance = cml.calcVariance(corrects, average)

    print(f'Statistics for {round(100*percent, 1)}% training data with {count} trials')
    print(f'\tAverage correct guesses: {average} out of {len(testLabels)} correctly')
    print(f'\tVariance: {variance}')
    print(f'\tStandard Deviation: {math.sqrt(variance)}\n')

    percent += 0.1

print(f'This took {round(time.time() - start_time, 2)} seconds')