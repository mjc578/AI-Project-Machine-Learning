import common_methods_lib as cml

import random
import math
import time

def getPercLabels(labelList, randRange):
    labtrix = []
    for n in randRange:
        labtrix.append(labelList[n])

    return labtrix

def predict(weight, featVects, z):
    sum = [0.0] * z
    for i in range(z):
        for j in range(len(featVects)):
            sum[i] += weight[i][j] * featVects[j]
        sum[i] += weight[i][len(featVects)]
    m = sum.index(max(sum))
    return m

#MAIN METHOD

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
    z = 2


elif which == 'digit':
    trainfn = 'digitdata/trainingimages'
    trainLabelsfn = 'digitdata/traininglabels'
    trainCount = 5000
    trainHeight = 28
    testfn = 'digitdata/testimages'
    testLabelsfn = 'digitdata/testlabels'
    testCount = 1000
    testHeight = 28
    z = 10

else:
    print('invalid input')
    exit(0)

#training data and labels
trainMatrix = cml.readInFile(trainfn, trainCount, trainHeight)
labelList = cml.readInLabels(trainLabelsfn)

#testing data, labels, and feature vectors
testingMatrix = cml.readInFile(testfn, testCount, testHeight)
testLabels = cml.readInLabels(testLabelsfn)
testFeatVects = cml.getFeatureVectors(testingMatrix, cml.featurePerPixel)

numWeights = len(testFeatVects[0]) + 1
  
#start the training data at 10%
percent = 0.1

entire_time = time.time()

while percent <= 1:
    
    currRange = int(round(percent, 1) * len(labelList))

    #need to average our results for each percent of training we take in, lets try 10 for now
    count = 0
    average = 0.0
    #keep track of # of times classifier guessed correctly for each trial to calculate variance and then sd
    corrects = []

    total_time = 0

    while(count < 5):
        #training begins here so stat timer
        start_time = time.time()
        #weight = [[0] * numWeights] * z
        weight = []
        for n in range(z):
            weight.append([])
            for nn in range(numWeights):
                weight[n].append(0)

        #get a list of numbers of currRange length which can range from 0 to number of labels/images
        randSamp = random.sample(range(len(labelList)), currRange)
       
        #get a whatever% sample of the label list and traintrix, importantly same indeces
        percOfLabList = getPercLabels(labelList, randSamp)
        percOfTraintrix = cml.randTrainImgs(trainMatrix, randSamp)

        #get the feature vectors for each image
        trainFeatVects = cml.getFeatureVectors(percOfTraintrix, cml.featurePerPixel)

        #Perceptron Algorithm
        totalError = 1
        epoch = 0
        while totalError != 0 and epoch != 1:
            totalError = 0
            epoch += 1
            for i in range(len(percOfLabList)):
                prediction = predict(weight, trainFeatVects[i], z)
                #print(f'This is the current label {percOfLabList[i]} and the prediction is {prediction}')
                if percOfLabList[i] != prediction:
                    #error = labelList[i] - prediction
                    for j in range(numWeights - 1):
                        weight[percOfLabList[i]][j] += trainFeatVects[i][j]
                        weight[prediction][j] -= trainFeatVects[i][j]
                    weight[percOfLabList[i]][numWeights - 1] += 1
                    weight[prediction][numWeights - 1] -= 1
                    totalError += 1

        #add the time it took to the total time to later get the average
        total_time += round(time.time() - start_time, 2)


        #COMMENCE GUESSAGE
        correctCount = 0
        for i in range(testCount):
            output = predict(weight, testFeatVects[i], z)
            if output == testLabels[i]:
                correctCount +=1
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
    print(f'\tStandard Deviation: {math.sqrt(variance)}')
    print(f'\tAverage training time for this percentage: {round(total_time/count, 2)} seconds\n')

    percent += 0.1

print(f'This entire process took {round(time.time() - entire_time, 2)} seconds')

