import numpy as np
import pandas as pd
from math import log2, prod
from operator import itemgetter


class DecisionTreeGenerator:

    def __init__(self, data):
        self.data = data
        self.numOfInputVars = data.iloc[0,0:-1].count()
        self.classes = set(data.iloc[:,-1])
        self.tree = None

    def checkIfOnlyOneClass(self, data):      
        distinctClasses = set(data.iloc[:,-1])           
        if len(distinctClasses) == 1:
            return (True, distinctClasses[0])
        else:
            return (False, None)

    def getNumberOfRecordsInEachClass(self, data):
        return [ (x, np.count_nonzero(data.iloc[:,-1].to_numpy() == x)) for x in self.classes ] 

    def getClassWithMostRecords(self, data):

        classes = set(data.iloc[:,-1])
        numOfRecordsInEachClass = self.getNumberOfRecordsInEachClass(data)
        return max(numOfRecordsInEachClass, key=itemgetter(1))[0]

    def calculateEntopy(self, data):
        denominator = len(data)
        numOfRecordsInEachClass = self.getNumberOfRecordsInEachClass(data)
        probabilities = [x[1]/denominator for x in numOfRecordsInEachClass]
        entropy = -sum([x * log2(x) for x in probabilities])
        return entropy

    def calculateInformationGain(self, data, dataSubsets):
        entropyBeforeSplit = self.calculateEntopy(data)

        denominator = len(data)  
        entropiesAfterSplit = [self.calculateEntopy(x) for x in dataSubsets]
        subsetProportions = [len(x)/denominator for x in dataSubsets ]
        weightedSumOfEntropies = -sum([x*y for x,y in zip(entropiesAfterSplit, subsetProportions)])
        
        informationGain = entropyBeforeSplit - weightedSumOfEntropies
        return informationGain

    def getNumberOfRecordsInClass(self, data, cls):
        numOfRecords = self.getNumberOfRecordsInEachClass(data)
        result = next((x for x in numOfRecords if x[0] == cls), (cls, 0))
        return result

    def calculateMeasureOfGoodness(self, data, dataSubsets):
        if len(dataSubsets) != 2:
            raise ValueError("Number of subsets is not 2")

        totalNumOfRecords = len(data)
        proportions = [len(x)/totalNumOfRecords for x in dataSubsets]

        numOfRecordsLeft = self.getNumberOfRecordsInEachClass(dataSubsets[0])
        numOfRecordsRight = self.getNumberOfRecordsInEachClass(dataSubsets[1])

        distances = [None] * len(self.classes)
        for i, cl in enumerate(self.classes):
            numOfClassRecordsLeft = self.getNumberOfRecordsInClass(dataSubsets[0], cl)[1]
            numOfClassRecordsRight = self.getNumberOfRecordsInClass(dataSubsets[1], cl)[1]

            distances[i] = abs(numOfClassRecordsLeft/totalNumOfRecords - numOfClassRecordsRight/totalNumOfRecords)

        return 2 * prod(proportions) * sum(distances)

           



class Node:

	def __init__(self, name):
		self.name = name
		self.childNodes = []   

