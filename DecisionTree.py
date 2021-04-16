import numpy as np
import pandas as pd
from math import log2, prod
from operator import itemgetter
from optbinning import OptimalBinning, MulticlassOptimalBinning
from ast import literal_eval
from collections import Counter

class DecisionTreeGenerator:

    def __init__(self, data):
        self.data = data
        self.inputVars = list(data)[:-1]
        # self.numOfInputVars = data.iloc[0,0:-1].count()
        self.classes = set(data.iloc[:, -1])
        # self.checkForCategoricalData()
        self.treeRoot = None

    def checkIfOnlyOneClass(self, data):
        '''Check if all entries in given dataframe belong to the same class, if true - return class name'''
        distinctClasses = set(data.iloc[:, -1])  # set containing all classes of the target varible
        if len(distinctClasses) == 1:
            return (True, distinctClasses.pop())
        else:
            return (False, None)

    def getNumberOfRecordsInEachClass(self, data):
        '''Returns a list of tuples (a,b) where: 
        a - class name (value of the target variable), 
        b - number of entries in dataframe, belonging to class a'''
        return [(x, np.count_nonzero(data.iloc[:, -1].to_numpy() == x)) for x in self.classes]

    def getClassWithMostRecords(self, data):
        '''Returns the name of the class with most records in the dataframe'''
        # classes = set(data.iloc[:,-1])
        numOfRecordsInEachClass = self.getNumberOfRecordsInEachClass(data)
        return max(numOfRecordsInEachClass, key=itemgetter(1))[0]

    def calculateEntopy(self, data):

        denominator = len(data)  # number of entries in data
        numOfRecordsInEachClass = self.getNumberOfRecordsInEachClass(data)
        probabilities = [x[1] / denominator for x in
                         numOfRecordsInEachClass]  # (number of entries in class a) / (total number of entries)
        entropy = -sum([x * (log2(x) if x > 0 else 0) for x in probabilities])  # by convention log2(0) = 0
        return entropy

    def calculateInformationGain(self, data, dataSubsets):
        entropyBeforeSplit = self.calculateEntopy(data)

        denominator = len(data)  # number of entries in data
        entropiesAfterSplit = [self.calculateEntopy(x) for x in dataSubsets]
        subsetProportions = [len(x) / denominator for x in dataSubsets]
        weightedSumOfEntropies = sum([x * y for x, y in zip(entropiesAfterSplit, subsetProportions)])

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
        proportions = [len(x) / totalNumOfRecords for x in dataSubsets]

        # numOfRecordsLeft = self.getNumberOfRecordsInEachClass(dataSubsets[0])
        # numOfRecordsRight = self.getNumberOfRecordsInEachClass(dataSubsets[1])

        distances = [None] * len(self.classes)
        for i, cl in enumerate(self.classes):
            numOfClassRecordsLeft = self.getNumberOfRecordsInClass(dataSubsets[0], cl)[1]
            numOfClassRecordsRight = self.getNumberOfRecordsInClass(dataSubsets[1], cl)[1]

            distances[i] = abs(numOfClassRecordsLeft / totalNumOfRecords - numOfClassRecordsRight / totalNumOfRecords)

        return 2 * prod(proportions) * sum(distances)

    # def checkForCategoricalData(self):
    #     '''Checks if the ratio between the number of unique and the total number of values for every input varialbe is lower than a user-defined threshold, if so - changes the type of the dataframe column to "category"'''
    #     for c in self.data.columns:
    #         if str(self.data[c].dtype) != 'category' and self.data[c].nunique()/self.data[c].count() < 0.05:
    #             self.data = self.data.astype({c:'category'})

    def splitData(self, data, availableAttributes, numericAttrBinning):
        '''Given a list of available attributes chooses a split that has 
        the largest information gain. Returns the chosen attribute, the 
        subsets of the dataframe resulting from the split, the best split 
        threshold and the ranges for each subset'''
        bestGain = -np.inf
        bestSubsets = None
        splitAttrib = None
        bestSplitThreshold = None
        bestRanges = None

        for attr in availableAttributes:
            # if attr is discrete attribute with z values
            if str(data[attr].dtype) == 'object' or str(data[attr].dtype) == 'category':
                          
                if len(set(data[attr])) == 1: continue # skip if only one category             
                grouped = data.groupby(attr)
                # get values for binning
                x = data[attr].values
                y = data.iloc[:, -1].astype('category').cat.codes.values

                optb = OptimalBinning(dtype='categorical', min_n_bins = 2, max_n_bins=4)
                optb.fit(x, y)
                binningResultDt = optb.binning_table.build()
                bins = binningResultDt['Bin'].head(-3)

                # create susbset for each bin if target var is binary and there are multiple bins 
                if(len(self.classes) == 2 and len(bins) > 1): # Binary targret variable
                    subsets = [pd.concat([grouped.get_group(cat) for cat in bin]) for bin in bins]

                else: # otherwise create subset for each value (category) of the attribute
                    subsets = [grouped.get_group(x) for x in data[attr].unique()]
                    
                infoGain = self.calculateInformationGain(data, subsets)

                if infoGain >= bestGain:
                    bestGain = infoGain
                    bestSubsets = subsets
                    splitAttrib = attr
                    bestSplitThreshold = None
                    bestRanges = None

            else:  # if attr has numeric values
                onlyOneBin = False
                # get values for binning
                x = data[attr].values
                y = data.iloc[:, -1].values

                optb = MulticlassOptimalBinning(min_n_bins=2, max_n_bins=4)
                optb.fit(x, y)
                binningResultDt = optb.binning_table.build()
                bins = binningResultDt['Bin'].head(-3)
                if len(bins) == 1:
                    onlyOneBin = True

                # if user enabled numeric attribue binning and there are multiple bins
                if numericAttrBinning is True and onlyOneBin is False:

                    # modify range string representation so it can be parsed
                    bins.iloc[0] = bins.iloc[0].replace('-inf', "'-inf'")
                    bins.iloc[-1] = bins.iloc[-1].replace('inf', "'inf'")
                    # create list of tuples for every range
                    ranges = [literal_eval(x.replace('[', '(')) for x in bins]
                    # replace 'inf' strigns with np.inf
                    ranges = [(-np.inf, x[1]) if x[0] == '-inf' else ((x[0], np.inf) if x[1] == 'inf' else (x[0], x[1])) for x in ranges]
                    # create subsets according to the ranges
                    subsets = [data.loc[(data[attr] >= r[0]) & (data[attr] < r[1])] for r in ranges]

                    infoGain = self.calculateInformationGain(data, subsets)

                    if infoGain >= bestGain:
                        bestGain = infoGain
                        bestSubsets = subsets
                        splitAttrib = attr
                        bestSplitThreshold = None
                        bestRanges = ranges                    
                else: # binary split using threshold                 
                    sortedData = data.sort_values(attr)  # sort data by attr
                    for i in range(len(sortedData[attr]) - 1):  # for each entry (without the last one)
                        # if current and next value of attr are equal - do nothing
                        if sortedData[attr].iloc[i] == sortedData[attr].iloc[i + 1]:
                            continue
                        # calculate threshold and use it to create two subsets
                        currentThreshold = (sortedData[attr].iloc[i] + sortedData[attr].iloc[i + 1]) / 2
                        lowerSubset = sortedData[sortedData[attr] <= currentThreshold]
                        higherSubset = sortedData[sortedData[attr] > currentThreshold]
                        infoGain = self.calculateInformationGain(sortedData, [lowerSubset, higherSubset])

                        if infoGain > bestGain:
                            bestGain = infoGain
                            bestSubsets = [lowerSubset, higherSubset]
                            splitAttrib = attr
                            bestSplitThreshold = currentThreshold
                    
        return (splitAttrib, bestSubsets, bestSplitThreshold, bestRanges)

    def generate(self, numericAttrBinning=True):
        '''Calls generateTree function for the dataframe assigned to the current instance of 
        DecisionTreeGenerator, considering all input variables. 
        Assigns the result to treeRoot data atribute of the instance.'''
        self.treeRoot = self.generateTree(self.data, self.inputVars, numericAttrBinning)

    def generateTree(self, data, availableAttributes, numericAttrBinning, dataRange=None ):
        '''Recursively generates a decision tree for given dataframe and 
        list of attributes. Returns an instance of class Node'''
        # Stopping criteria
        checkIfOnlyOneClass = self.checkIfOnlyOneClass(data)
        if checkIfOnlyOneClass[0] is True:
            return Node(name=checkIfOnlyOneClass[1], threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        if len(availableAttributes) == 0:
            clasWithMostRecords = self.getClassWithMostRecords(data)
            return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        if len(data) < 1:
            clasWithMostRecords = self.getClassWithMostRecords(data)
            return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        # if more than 90% of the records in data belong to the same class
        if (Counter(data.iloc[:, -1].values).most_common(1)[0][1] / len(data)) * 100 > 90:
            clasWithMostRecords = self.getClassWithMostRecords(data)
            return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        splitResult = self.splitData(data, availableAttributes, numericAttrBinning)
        remainingAvailableAttribues = availableAttributes.copy()

        # if no suitable attribute to split on
        if splitResult[0] is None: return

        #if splitResult[2] is None: # If attr is categorical, remove it from available attributes
        remainingAvailableAttribues.remove(splitResult[0])

        decisionNode = Node(name=splitResult[0], threshold=splitResult[2], isLeafNode=False, data=data, dataRange=dataRange)
        # Recursive call for all subsets resulting from the split
        if splitResult[3] is None:          
            decisionNode.childNodes = [self.generateTree(subset, remainingAvailableAttribues, numericAttrBinning) for subset in splitResult[1]]
        else: # if the split was done with binning, send the range to the next recursive call
            decisionNode.childNodes = [self.generateTree(subset, remainingAvailableAttribues, numericAttrBinning, dataRange=dataRng) for (subset, dataRng) in zip(splitResult[1], splitResult[3])]

        # Remove child node elements of type None, generated if split was not possible
        decisionNode.childNodes = [node for node in decisionNode.childNodes if node is not None]
        return decisionNode


class Node:

    def __init__(self, name, threshold, isLeafNode, data, dataRange=None):
        self.name = name
        self.threshold = threshold
        self.isLeafNode = isLeafNode
        self.data = data
        self.dataRange = dataRange
        self.childNodes = []

    def print(self, indentaion='', file=None):
        '''Traverses all child nodes and recursively prints info about them in a tree-like fashion'''

        # if the node is a Leaf node printing is handled by the parent
        if self.isLeafNode:
            return

        print(indentaion + '=' * 10 + ' ' + self.name + ' ' + '=' * 10, file=file)

        for index, childNode in enumerate(self.childNodes):
            if self.threshold is None:
                if childNode.dataRange is None:
                    if childNode.isLeafNode:
                        print(indentaion + str(self.name) + ' ' + str(childNode.data.loc[:, self.name].unique().tolist()) + ' [size = ' + str(len(childNode.data)) + ']' + ": " + str(childNode.name), file=file)
                    else:
                        print(indentaion + str(self.name) + ' ' + str(childNode.data.loc[:, self.name].unique().tolist()) + ' [size = ' + str(len(childNode.data)) + ']' + ":", file=file)
                        childNode.print(indentaion + '\t', file=file)
                else:
                    if childNode.isLeafNode:
                        print(indentaion + str(self.name) + ' \u2208 ' + str(childNode.dataRange) + ' [size = ' + str(len(childNode.data)) + ']' + ": " + str(childNode.name), file=file)
                    else:
                        print(indentaion + str(self.name) + ' \u2208 ' + str(childNode.dataRange) + ' [size = ' + str(len(childNode.data)) + ']' + ":", file=file)
                        childNode.print(indentaion + '\t', file=file)
            else:
                if childNode.isLeafNode:
                    print(indentaion + str(self.name) + " [" + ['<= ', '> '][index] + str(
                        self.threshold) + "]" + ' [size = ' + str(len(childNode.data)) + ']' + ": " + str(childNode.name), file=file)
                else:
                    print(indentaion + str(self.name) + " [" + ['<= ', '> '][index] + str(self.threshold) + "]" + ' [size = ' + str(len(childNode.data)) + ']' + ":", file=file)
                    childNode.print(indentaion + '\t', file=file)
