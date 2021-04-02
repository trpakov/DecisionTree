import numpy as np
import pandas as pd
from math import log2, prod
from operator import itemgetter


class DecisionTreeGenerator:

    def __init__(self, data):
        self.data = data
        self.inputVars = list(data)[:-1]
        # self.numOfInputVars = data.iloc[0,0:-1].count()
        self.classes = set(data.iloc[:,-1])
        # self.checkForCategoricalData()
        self.treeRoot = None

    def checkIfOnlyOneClass(self, data):
        '''Check if all entries in given dataframe belong to the same class, if true - return class name'''       
        distinctClasses = set(data.iloc[:,-1]) # set containing all classes of the target varible           
        if len(distinctClasses) == 1:
            return (True, distinctClasses.pop())
        else:
            return (False, None)

    def getNumberOfRecordsInEachClass(self, data):
        '''Returns a list of tuples (a,b) where: 
        a - class name (value of the target variable), 
        b - number of entries in dataframe, belonging to class a'''
        return [ (x, np.count_nonzero(data.iloc[:,-1].to_numpy() == x)) for x in self.classes ] 

    def getClassWithMostRecords(self, data):
        '''Returns the name of the class with most records in the dataframe'''
        classes = set(data.iloc[:,-1])
        numOfRecordsInEachClass = self.getNumberOfRecordsInEachClass(data)
        return max(numOfRecordsInEachClass, key=itemgetter(1))[0]

    def calculateEntopy(self, data):
    
        denominator = len(data) # number of entries in data
        numOfRecordsInEachClass = self.getNumberOfRecordsInEachClass(data)
        probabilities = [x[1]/denominator for x in numOfRecordsInEachClass] # (number of entries in class a) / (total number of entries)
        entropy = -sum([x * (log2(x) if x>0 else 0) for x in probabilities]) # by convention log2(0) = 0
        return entropy

    def calculateInformationGain(self, data, dataSubsets):
        entropyBeforeSplit = self.calculateEntopy(data)

        denominator = len(data) # number of entries in data
        entropiesAfterSplit = [self.calculateEntopy(x) for x in dataSubsets]
        subsetProportions = [len(x)/denominator for x in dataSubsets ]
        weightedSumOfEntropies = sum([x*y for x,y in zip(entropiesAfterSplit, subsetProportions)])

        informationGain = entropyBeforeSplit - weightedSumOfEntropies
        return informationGain

    def getNumberOfRecordsInClass(self, data, cls):
        '''Returns the number of entries in the dataframe, belonging to the specified class name'''
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

    # def checkForCategoricalData(self):
    #     '''Checks if the ratio between the number of unique and the total number of values for every input varialbe is lower than a user-defined threshold, if so - changes the type of the dataframe column to "category"'''
    #     for c in self.data.columns:
    #         if str(self.data[c].dtype) != 'category' and self.data[c].nunique()/self.data[c].count() < 0.05:
    #             self.data = self.data.astype({c:'category'})

    def splitData(self, data, availableAttributes):
        '''Given a list of available attributes chooses a split that has 
        the largest information gain. Returns the chosen attribute and the 
        subsets of the dataframe resulting from the split'''
        bestGain = -np.inf
        bestSubsets = None
        splitAttrib = None
        bestSplitThreshold = None

        for attr in availableAttributes:
            # if attr is discrete attribute with z values
            if str(data[attr].dtype) == 'object' or str(data[attr].dtype) == 'category':
                grouped = data.groupby(attr)
                subsets = [grouped.get_group(x) for x in data[attr].unique()] # one subset for each value
                infoGain = self.calculateInformationGain(data, subsets)

                if infoGain >= bestGain:
                    bestGain = infoGain
                    bestSubsets = subsets
                    splitAttrib = attr
                    bestSplitThreshold = None
           
            else: # if attr has numeric values
                sortedData = data.sort_values(attr) # sort data by attr
                for i in range(len(sortedData[attr]) - 1): # for each entry (without the last one)
                    # if current and next value of attr are equal - do nothing
                    if sortedData[attr].iloc[i] == sortedData[attr].iloc[i+1]:
                        continue
                    
                    # calculate threshold and use it to create two subsets
                    currentThreshold = (sortedData[attr].iloc[i] + sortedData[attr].iloc[i+1]) / 2
                    lowerSubset = sortedData[sortedData[attr] <= currentThreshold]
                    higherSubset = sortedData[sortedData[attr] > currentThreshold]
                    infoGain = self.calculateInformationGain(sortedData, [lowerSubset, higherSubset])
                    
                    if infoGain > bestGain:
                        bestGain = infoGain
                        bestSubsets = [lowerSubset, higherSubset]
                        splitAttrib = attr
                        bestSplitThreshold = currentThreshold
                    
        return (splitAttrib, bestSubsets, bestSplitThreshold)

    def generate(self):
        '''Calls generateTree function for the dataframe assigned to the current instance of 
        DecisionTreeGenerator, considering all input variables. 
        Assigns the result to treeRoot data atribute of the instance.'''
        self.treeRoot = self.generateTree(self.data, self.inputVars)

    def generateTree(self, data, availableAttributes):
        '''Recursively generates a decision tree for given dataframe and 
        list of attributes. Returns an instance of class Node'''

        # Stopping criteria
        checkIfOnlyOneClass = self.checkIfOnlyOneClass(data)
        if checkIfOnlyOneClass[0] is True:
            return Node(name=checkIfOnlyOneClass[1], threshold=None, isLeafNode=True, data=data)
        
        if len(availableAttributes) == 0:
            clasWithMostRecords = self.getClassWithMostRecords(data)
            return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data)

        splitResult = self.splitData(data, availableAttributes)
        remainingAvailableAttribues = availableAttributes.copy()
        remainingAvailableAttribues.remove(splitResult[0])

        decisionNode = Node(name=splitResult[0], threshold=splitResult[2], isLeafNode=False, data=data)
        # Recursive call for all subsets resulting from the split
        decisionNode.childNodes = [self.generateTree(subset, remainingAvailableAttribues) for subset in splitResult[1]]

        return decisionNode


class Node:

    def __init__(self, name, threshold, isLeafNode, data):
        self.name = name
        self.threshold = threshold
        self.isLeafNode = isLeafNode
        self.data = data
        self.childNodes = []

    def print(self, indentaion=''):
        '''Traverses all child nodes and recursively prints info about them in a tree-like fashion'''

        # if the node is a Leaf node printing is handled by the parent
        if self.isLeafNode: 
            return
        
        print(indentaion + '='*6 + ' ' + self.name + ' ' + '='*6)
        
        for index, childNode in enumerate(self.childNodes):       
            if self.threshold is None:        
                if childNode.isLeafNode:
                    print(indentaion + self.name + " [" + childNode.data.loc[:, self.name].tolist()[0] + "]: " + childNode.name)
                else:
                    print(indentaion + self.name + " [" + childNode.data.loc[:, self.name].tolist()[0] + "]:")
                    childNode.print(indentaion + '\t')
            else:
                if childNode.isLeafNode:
                    print(indentaion + self.name + " [" + ['<= ', '> '][index] + str(self.threshold) + "]: " + childNode.name)
                else:
                    print(indentaion + self.name + " [" + ['<= ', '> '][index] + str(self.threshold) + "]:")
                    childNode.print(indentaion + '\t')


