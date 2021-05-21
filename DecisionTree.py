import numpy as np
import pandas as pd
from math import log2, prod
from operator import itemgetter
from optbinning import OptimalBinning, MulticlassOptimalBinning, ContinuousOptimalBinning
from ast import literal_eval
from collections import Counter

class DecisionTreeGenerator:
    '''Main class for tree generation, pruning and fitting.'''

    def __init__(self, data):
        self.data = data
        self.inputVars = list(data)[:-1]
        # self.numOfInputVars = data.iloc[0,0:-1].count()
        self.classes = set(data.iloc[:, -1])
        # self.checkForCategoricalData()
        self.treeRoot = None
        self.numOfLeafNodes = 0 # number of terminal nodes
        self.twigs = {} # dict with nodes that have only leaf nodes (keys) and their info gain (values)
        self.numericAttrRanges = {x: [[[(-np.inf, np.inf)]], 0] for x in self.inputVars} # dict with parent ranges for repeating numeric attributes, used when repeatAttributes is True

        # set tree type
        if str(data.iloc[:, -1].dtype) == 'object' or str(data.iloc[:, -1].dtype) == 'category':
            self.treeType = 'classification'
        else:
            self.treeType = 'regression'

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

    def calculateStandardDeviationReduction(self, data, dataSubsets):
        standartDeviations = [np.std(subset.iloc[:, -1]) for subset in dataSubsets]
        probabilities = [len(subset)/len(data) for subset in dataSubsets]
        stdAfterSplit = sum(x * y for x, y in zip(probabilities, standartDeviations))
        stdBeforeSplit = np.std(data.iloc[:, -1])
        return stdBeforeSplit - stdAfterSplit

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

    def checkForCategoricalData(self, threshold=0.05):
        '''Checks if the ratio between the number of unique values and the total 
        number of values for every attribute is lower than a user-defined 
        threshold, if so - changes the type of the dataframe column to "category".'''
        for c in self.data.columns:
            if str(self.data[c].dtype) != 'category' and self.data[c].nunique()/self.data[c].count() < threshold:
                self.data = self.data.astype({c:'category'})

    def splitData(self, data, availableAttributes, numericAttrBinning, repeatAttributes, minNumRecordsLeafNode):
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
                y = data.iloc[:, -1].values

                # type of binning is determined by tree type
                if self.treeType == 'classification':
                    optb = OptimalBinning(dtype='categorical', min_n_bins = 2, max_n_bins=4)
                else:                  
                    optb = ContinuousOptimalBinning(dtype='categorical', min_n_bins = 2, max_n_bins=4, min_prebin_size=0.001)
                    
                optb.fit(x, y)
                binningResultDt = optb.binning_table.build()
                bins = binningResultDt['Bin'].head(-3)

                # create susbset for each bin if target var is binary and there are multiple bins 
                if(len(self.classes) == 2 and len(bins) > 1): # Binary targret variable
                    subsets = [pd.concat([grouped.get_group(cat) for cat in bin]) for bin in bins]

                else: # otherwise create subset for each value (category) of the attribute
                    subsets = [grouped.get_group(x) for x in data[attr].unique()]

                if any(len(subset) for subset in subsets if len(subset) < minNumRecordsLeafNode):
                    continue # skip if there are too small subsets

                if self.treeType == 'classification':
                    infoGain = self.calculateInformationGain(data, subsets)
                else:
                    infoGain = self.calculateStandardDeviationReduction(data, subsets)

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

                # type of binning is determined by tree type
                if self.treeType == 'classification':
                    optb = MulticlassOptimalBinning(min_n_bins=2, max_n_bins=4)
                else:
                    if x.min() == x.max(): continue
                    optb = ContinuousOptimalBinning(min_n_bins=2, max_n_bins=4,min_prebin_size=0.001)
                
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

                    if any(len(subset) for subset in subsets if len(subset) < minNumRecordsLeafNode):
                        continue # skip if there are too small subsets

                    if self.treeType == 'classification':
                        infoGain = self.calculateInformationGain(data, subsets)
                    else:
                        infoGain = self.calculateStandardDeviationReduction(data, subsets)

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

                        if len(lowerSubset) < minNumRecordsLeafNode or len(higherSubset) < minNumRecordsLeafNode:
                            continue # skip if there are too small subsets

                        if self.treeType == 'classification':
                            infoGain = self.calculateInformationGain(data, [lowerSubset, higherSubset])
                        else:
                            infoGain = self.calculateStandardDeviationReduction(data, [lowerSubset, higherSubset])

                        if infoGain > bestGain:
                            bestGain = infoGain
                            bestSubsets = [lowerSubset, higherSubset]
                            splitAttrib = attr
                            bestSplitThreshold = currentThreshold
                            bestRanges = None

        # fix ranges if repeatingAttributes
        if bestRanges and repeatAttributes:
            parentRanges = self.numericAttrRanges[splitAttrib][0][self.numericAttrRanges[splitAttrib][1]]
            checkValue = data[splitAttrib].iloc[0]
            parentRange = next(rng for rng in parentRanges if checkValue >= rng[0] and checkValue < rng[1])
            bestRanges[0] = (parentRange[0], bestRanges[0][1])
            bestRanges[-1] = (bestRanges[-1][0], parentRange[-1])
            
            self.numericAttrRanges[splitAttrib][1] += 1

            if self.numericAttrRanges[splitAttrib][1] in range(0, len(self.numericAttrRanges[splitAttrib][0])):
                self.numericAttrRanges[splitAttrib][0][self.numericAttrRanges[splitAttrib][1]] = bestRanges
            else:
                self.numericAttrRanges[splitAttrib][0].append(bestRanges)

        return (splitAttrib, bestSubsets, bestSplitThreshold, bestRanges, bestGain)

    def generate(self, numericAttrBinning=True, minNumRecordsLeafNode=10, repeatAttributes=False):
        '''Calls generateTree function for the dataframe assigned to the current instance of 
        DecisionTreeGenerator, considering all input variables. 
        Assigns the result to treeRoot data atribute of the instance.'''
        self.treeRoot = self.generateTree(self.data, self.inputVars, numericAttrBinning, minNumRecordsLeafNode=minNumRecordsLeafNode, repeatAttributes=repeatAttributes)

    def generateTree(self, data, availableAttributes, numericAttrBinning, dataRange=None, minNumRecordsLeafNode=10, repeatAttributes=False):
        '''Recursively generates a decision tree for given dataframe and 
        list of attributes. Returns an instance of class Node'''
        # Stopping criteria
        checkIfOnlyOneClass = self.checkIfOnlyOneClass(data)
        if checkIfOnlyOneClass[0] is True:
            self.numOfLeafNodes += 1
            return Node(name=checkIfOnlyOneClass[1], threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        if len(availableAttributes) == 0:
            self.numOfLeafNodes += 1
            if self.treeType == 'classification':
                clasWithMostRecords = self.getClassWithMostRecords(data)
                return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)
            else:
                averageValue = np.mean(data.iloc[:, -1])
                return Node(name=averageValue, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        # if len(data) <= maxNumRecordsToSkipSplitting:
        #     self.numOfLeafNodes += 1
        #     clasWithMostRecords = self.getClassWithMostRecords(data)
        #     return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        if self.treeType == 'classification':
            # if more than 90% of the records in data belong to the same class
            if (Counter(data.iloc[:, -1].values).most_common(1)[0][1] / len(data)) * 100 > 90:
                self.numOfLeafNodes += 1
                clasWithMostRecords = self.getClassWithMostRecords(data)
                return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)
        else:
            # if regression and coefficent of deviation (CV) less than 10%
            if (np.std(data.iloc[:, -1]) / np.mean(data.iloc[:, -1])) * 100 < 10:
                self.numOfLeafNodes += 1
                averageValue = np.mean(data.iloc[:, -1])
                return Node(name=averageValue, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        splitResult = self.splitData(data, availableAttributes, numericAttrBinning, repeatAttributes, minNumRecordsLeafNode)
        remainingAvailableAttribues = availableAttributes.copy()

        # if no suitable attribute to split on
        if splitResult[0] is None: 
            self.numOfLeafNodes += 1

            if self.treeType == 'classification':
                clasWithMostRecords = self.getClassWithMostRecords(data)
                return Node(name=clasWithMostRecords, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)
            else:
                averageValue = np.mean(data.iloc[:, -1])
                return Node(name=averageValue, threshold=None, isLeafNode=True, data=data, dataRange=dataRange)

        if repeatAttributes is False: # Remove current attribute from available attributes if user does not want repetition
            remainingAvailableAttribues.remove(splitResult[0])

        decisionNode = Node(name=splitResult[0], threshold=splitResult[2], isLeafNode=False, data=data, dataRange=dataRange, gain=splitResult[4])
        # Recursive call for all subsets resulting from the split
        if splitResult[3] is None:          
            decisionNode.childNodes = [self.generateTree(subset, remainingAvailableAttribues, numericAttrBinning, minNumRecordsLeafNode=minNumRecordsLeafNode, repeatAttributes=repeatAttributes) for subset in splitResult[1]]
        else: # if the split was done with binning, send the range to the next recursive call
            decisionNode.childNodes = [self.generateTree(subset, remainingAvailableAttribues, numericAttrBinning, dataRange=dataRng, minNumRecordsLeafNode=minNumRecordsLeafNode, repeatAttributes=repeatAttributes) for (subset, dataRng) in zip(splitResult[1], splitResult[3])]

        if repeatAttributes and splitResult[3]:
            self.numericAttrRanges[splitResult[0]][1] -= 1
        return decisionNode

    def generateTwigsDict(self, node=None):
        ''' Fill in the dict containing all current twigs (nodes whose children are all leaf nodes)'''
        if node is None: # if node is not specified, start from root and clear current dict
            node = self.treeRoot
            self.twigs.clear()

        if node.isLeafNode: # recursion stoppping criterium
            return

        # if all children are leaf nodes, add current node and its gain to dict
        if all(childNode.isLeafNode for childNode in node.childNodes):
            self.twigs[node] = node.gain

        # recursive call for all children of current node
        for node in node.childNodes:
            self.generateTwigsDict(node)

    def prune(self, maxNumOfLeafNodes):  
        '''Prunes out portions of the tree that result in the least information gain.
        Continues while number of leaf nodes is more than desired.'''
        
        while self.numOfLeafNodes > maxNumOfLeafNodes:
            self.generateTwigsDict() # fill in twigs dict
            # get twig node with lowest gain
            nodeWithLowestGain = min(self.twigs, key=self.twigs.get)
            # change type to leaf
            nodeWithLowestGain.isLeafNode = True
            # update total num of leaf nodes
            self.numOfLeafNodes -= (len(nodeWithLowestGain.childNodes) - 1)
            nodeWithLowestGain.gain = None
            nodeWithLowestGain.childNodes = [] # no more child nodes
            # change node name
            if self.treeType == 'classification':
                nodeWithLowestGain.name = self.getClassWithMostRecords(nodeWithLowestGain.data)
            else:
                nodeWithLowestGain.name = np.mean(nodeWithLowestGain.data.iloc[:, -1])
            #print('num of leaf nodes: ' + str(self.numOfLeafNodes))

    def traverseTree(self, row, node=None):
        '''Recursively traverses a structure of Nodes using values in 
        dataframe row for navigation. Returns the class corresponding 
        to the leaf node that was reached.'''
        if node is None: # if node not specified, start from root
            node = self.treeRoot

        if node.isLeafNode: # if node is leaf, return its name (class of the target variable)
            return node.name

        if node.threshold: # if binary split using threshold
            if row[node.name] <= node.threshold: # find correct child node
                return self.traverseTree(row, node.childNodes[0])
            else:
                return self.traverseTree(row, node.childNodes[1])

        if any(child for child in node.childNodes if child.dataRange): # if split into ranges using binning
            return self.traverseTree(row, next(child for child in node.childNodes if row[node.name] >= child.dataRange[0] and row[node.name] < child.dataRange[1]))
        else:
            return self.traverseTree(row, next(child for child in node.childNodes if row[node.name] in child.data.loc[:, node.name].unique().tolist()))

    def classify(self, data):
        '''Classify records in given dataset using already constructed tree.'''
        if self.treeRoot is None: # if tree not generated
            raise ValueError('Generate tree before classifying')

        if list(data) != self.inputVars: # if different attributes
            raise ValueError('Test dataframe is different from training dataframe')

        data[list(self.data)[-1]] = None # add empty column for target variable

        for i, row in data.iterrows(): # for each row in test data
            data.at[i, list(self.data)[-1]] = self.traverseTree(row) # set classification


class Node:
    '''Class containing information for a single tree node.'''

    def __init__(self, name, threshold, isLeafNode, data, dataRange=None, gain=None):
        self.name = name
        self.threshold = threshold
        self.isLeafNode = isLeafNode
        self.data = data
        self.dataRange = dataRange
        self.gain = gain
        self.childNodes = []

    def print(self, indentation='', numOfTabsBetweenLevels=1, file=None):
        '''Traverses all child nodes and recursively prints info about them in a tree-like fashion'''

        # if the node is a Leaf node printing is handled by the parent
        if self.isLeafNode:
            return

        print(indentation + '=' * 10 + ' ' + self.name + ' ' + '=' * 10, file=file)

        for index, childNode in enumerate(self.childNodes):
            if self.threshold is None:
                if childNode.dataRange is None:
                    if childNode.isLeafNode:
                        print(indentation + str(self.name) + ' ' + str(childNode.data.loc[:, self.name].unique().tolist()) + ' [size = ' + str(len(childNode.data)) + ']' + ": " + (str(childNode.name) if str(childNode.data.iloc[:, -1].dtype) in ['object', 'category'] else str(round(childNode.name, 3))) + (' [' + str(round(100 * len(childNode.data.loc[childNode.data.iloc[:, -1] == childNode.name])/len(childNode.data), 2)) + '%]' if str(childNode.data.iloc[:, -1].dtype) in ['object', 'category'] else ''), file=file)
                    else:
                        print(indentation + str(self.name) + ' ' + str(childNode.data.loc[:, self.name].unique().tolist()) + ' [size = ' + str(len(childNode.data)) + ']' + ":", file=file)
                        childNode.print(indentation + '\t' * numOfTabsBetweenLevels, numOfTabsBetweenLevels, file=file)
                else:
                    if childNode.isLeafNode:
                        print(indentation + str(self.name) + ' \u2208 ' + str(childNode.dataRange) + ' [size = ' + str(len(childNode.data)) + ']' + ": " + (str(childNode.name) if str(childNode.data.iloc[:, -1].dtype) in ['object', 'category'] else str(round(childNode.name, 3))) + (' [' + str(round(100 * len(childNode.data.loc[childNode.data.iloc[:, -1] == childNode.name])/len(childNode.data), 2)) + '%]' if str(childNode.data.iloc[:, -1].dtype) in ['object', 'category'] else ''), file=file)
                    else:
                        print(indentation + str(self.name) + ' \u2208 ' + str(childNode.dataRange) + ' [size = ' + str(len(childNode.data)) + ']' + ":", file=file)
                        childNode.print(indentation + '\t' * numOfTabsBetweenLevels, numOfTabsBetweenLevels, file=file)
            else:
                if childNode.isLeafNode:
                    print(indentation + str(self.name) + " [" + ['<= ', '> '][index] + str(
                        self.threshold) + "]" + ' [size = ' + str(len(childNode.data)) + ']' + ": " + (str(childNode.name) if str(childNode.data.iloc[:, -1].dtype) in ['object', 'category'] else str(round(childNode.name, 3))) + (' [' + str(round(100 * len(childNode.data.loc[childNode.data.iloc[:, -1] == childNode.name])/len(childNode.data), 2)) + '%]' if str(childNode.data.iloc[:, -1].dtype) in ['object', 'category'] else ''), file=file)
                else:
                    print(indentation + str(self.name) + " [" + ['<= ', '> '][index] + str(self.threshold) + "]" + ' [size = ' + str(len(childNode.data)) + ']' + ":", file=file)
                    childNode.print(indentation + '\t' * numOfTabsBetweenLevels, numOfTabsBetweenLevels, file=file)
