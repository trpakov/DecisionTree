# DecisionTree

DecisionTree is a variat of C4.5 decision tree algorithm implemented in Python. Developed as an assignment in Applied Artificial Intelligence course.

## Features

* Supports both categorical and numerical attributes.
* Uses information gain to determine the best split.
* Utilizes [OptBinning](https://github.com/guillermo-navas-palencia/optbinning) to find optimal binning for given attribute.
* Prunes the tree until leaf nodes are fewer than specified threshold.
* Classifies test data based on tree generated from tarining data.
* Provides additional methods for data preparation.

## Installation

Clone git repository:

```bash
git clone https://github.com/trpakov/DecisionTree.git
```

Use [pip](https://pip.pypa.io/en/stable/) to install required dependencies:

```bash
pip install -r requirements.txt
```

## Example usage

```python
import DecisionTree as dt
import pandas as pd
```

### Tree generaion

```python
# import training data from csv file
data = pd.read_csv("data/titanic2.csv")
# set column type as appropriate
data = data.astype({'Pclass':'category', 'Siblings/Spouses Aboard':'category', 'Parents/Children Aboard':'category'}, copy=False)
# create instance
dtg = dt.DecisionTreeGenerator(data)
# generate tree
dtg.generate()
```

### Tree pruning

```python
# prune tree so the leaf nodes are no more than 10
dtg3.prune(10)
```
```python
========== Sex ==========
Sex ['male'] [size = 573]:
	========== Age ==========
	Age ∈ (-inf, 13.0) [size = 41]:
		========== Siblings/Spouses Aboard ==========
		Siblings/Spouses Aboard [3, 5, 8, 4] [size = 17]: 0
		Siblings/Spouses Aboard [0] [size = 7]: 1
		Siblings/Spouses Aboard [1, 2] [size = 17]: 1
	Age ∈ (13.0, 20.25) [size = 89]: 0
	Age ∈ (20.25, 22.5) [size = 46]: 0
	Age ∈ (22.5, inf) [size = 397]: 0
Sex ['female'] [size = 314]:
	========== Pclass ==========
	Pclass [3] [size = 143]: 0
	Pclass [2] [size = 76]: 1
	Pclass [1, 4] [size = 95]: 1
```
### Saving tree to file

```python
# print generated tree to file
with open('data/titanic.txt', mode='w', encoding='utf-8') as f:
    dtg3.treeRoot.print(file=f)
```

### Classifying new data

```python
# import test data from csv file
testData = pd.read_csv("data/TitanicTest.csv")
# classify test data using generated tree
dtg.classify(testData)
# save results to csv file
testData.to_csv('data/TitanicResult.csv', index=False)
``` 

## Tree building algorithm outline

1. Check if termination criteria are satisfied.

2. Compute highest information gain for each attribute.

3. Choose the attribute whose split produces the best gain.

4. Create a decision node based on attribute chosen in step 3.

5. Split the dataset based on node from step 4.

6. Recursively call the algorithm for all subsets resulting from the split in step 5.

7. Attach new branches of the tree from step 6 to decision node from step 4.

8. Return generated tree.

## Pseudocode

```vba
Tree function GenerateTree(dataset D)
    {
        Tree = {}
        if (D is "pure" OR other stopping criteria met):
            terminate
        end if

        for all attribute a in D do:
            Compute information gain if we split on a
        end for

        a_best = Best attribute according to above computed criteria
        Tree = Create a decision node that splits on a_best in the root
        D_s = Subsets from D based on a_best

        for all D_s do:
            Tree_s = GenerateTree(D_s)
            Attach Tree_s to the corresponding branch of Tree
        end for

        return Tree
    }
```

## Documentation

### DecisionTreeGenerator(data)

* Main class accepting input data and generating a decision tree.

#### Parameters

* `data (pandas.DataFrame)` - Training data for tree generation.

#### Properties

* `data (pandas.DataFrame)` - Training data for tree generation.
* `inputVars (list)` - List of attributes.
* `classes (set)` - Set of classes (possible values of the target variable).
* `treeRoot (Node)` - Root node of the tree.
* `numOfLeafNodes (int)` - Total number of leaf nodes.
* `twigs (dict(Node))` - Dictionary with twig nodes (used as keys) whose children are only leaf nodes and their information gain (used as values).
* `numericAttrRanges (dict)` - Dictionary with previous intervals for repeating numeric attributes. Used when `repeatAttributes` is `True`

#### Methods
* `checkIfOnlyOneClass(data)` - Checks if all entries in given dataframe belong to the same class.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to check.

    ##### Returns
    * `(Bool, className)` - Tuple with first element `True` and second element - class name if only one class, otherwise `False` and `None`.
    
* `getNumberOfRecordsInEachClass(data)` - Finds and returns the number of records belonging to each class.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.

    ##### Returns
    * `list((a,b))` - List of tuples `(a, b)` where a - class name (value of target variable), b - number of entries in `data` with this class.

* `getClassWithMostRecords(data)` - Finds and returns the class that corresponds to most of the records.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.

    ##### Returns
    * `className (str)` - Name of tha class with most records in the dataframe.

* `getNumberOfRecordsInClass(data, cls)` - Finds and returns the number of entries in the dataframe, belonging to the specified class name.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.
    * `cls (str)` - Class name to use.

    ##### Returns
    * `result (int)` - Number of records in `data` which have class `cls`.

* `calculateEntropy(data)` - Calculates the information entropy for given dataframe.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.

    ##### Returns
    * `entropy (float)` - Information entropy for `data`. 
    
* `calculateInformationGain(data, dataSubsets)` - Calculates the gain information entropy after splitting given dataframe into subsets.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.
    * `dataSubsets list((pandas.DataFrame))` - Dataframes resulting from a split.

    ##### Returns
    * `informationGain (float)` - Increase in information produced by partitioning `data` into `dataSubsets`.

* `calculateMeasureOfGoodness(data, dataSubsets)` - Calculates a value that seeks to optimize the balance of a candidate split's capacity to create pure children with its capacity to create equally-sized children.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.
    * `dataSubsets list((pandas.DataFrame))` - Dataframes resulting from a split.

    ##### Returns
    * `measureOfGoodness (float)` - Calculated value for input `data` split into `dataSubsets`.

 * `checkForCategoricalData(threshold)` - Heuristic method that checks if the ratio between the number of unique values and the total number of values for every attribute is lower than a user-defined threshold, if so - changes the type of the dataframe column to "category".

    ##### Parameters
    * `threshold (float) (default=0.05)` - Threshold to use. Column type is changed if the ratio between the number of unique values and the total number of values for every attribute is lower than this value.

    ##### Returns
    * `None`  
    
* `splitData(data, availableAttributes, numericAttrBinning, repeatAttributes, minNumRecordsLeafNode)` - Given a list of available attributes chooses a split that has the largest information gain.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.
    * `availableAttributes (list)` - List of attribute names considered for splitting on.
    * `numericAttrBinning (bool)` - If `True` - numeric attributes are split using binning, if `False` - binary split is performed.
    * `repeatAttributes (bool)` - If `True` - multiple splits can be performed over the same attribute, if `False` - once an attribute is part of a split it is no longer considered in further splits.
    * `minNumRecordsLeafNode (int)` - Minimum number of records in a leaf node. A split is ignored if there is a subset with lower number of elements than this value.

    ##### Returns
    * `splitResult (splitAttrib, bestSubsets, bestSplitThreshold, bestRanges, bestGain)` - Tuple containing information for the split performed:
        * `splitAttrib (str)` - Name of chosen attribute.  
        * `bestSubsets (list(pandas.DataFrame))` - List of dataframes after splitting `data`.  
        * `bestSplitThreshold (float)` - Number indicating the split point of a binary split.  
        * `bestRanges (list(float, float))` - List with tuples corresponding to the ranges produced after binning.  
        * `bestGain (float)` - Information gain value for the chosen split.

* `generate(numericAttrBinning, minNumRecordsLeafNode, repeatAttributes)` - Calls `generateTree` function for the dataframe assigned to the current instance of `DecisionTreeGenerator`, considering all input variables. Assigns the result to `treeRoot` attribute of the instance.

    ##### Parameters
    * `numericAttrBinning (bool) (default=True)` - If `True` - numeric attributes are split using binning, if `False` - binary split is performed.
    * `maxNumRecordsToSkipSplitting (int) (default=30)` - Directly return a leaf node if number of records is lower than this value. Prevents overfitting.
    * `minNumRecordsLeafNode (int) (default=10)` - Minimum number of records in a leaf node. A split is ignored if there is a subset with lower number of elements than this value.
    * `repeatAttributes (bool) (default=False)` - If `True` - multiple splits can be performed over the same attribute, if `False` - once an attribute is part of a split it is no longer considered in further splits.

    ##### Returns
    * `None`

* `generateTree(data, availableAttributes, numericAttrBinning, dataRange, minNumRecordsLeafNode, repeatAttributes)` - Called by `generate` function. Generates the tree by recursively calling itself, assigning child nodes to decision nodes and returning leaf nodes when stopping criteria are met.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.
    * `availableAttributes (list)` - List of attribute names considered for splitting on.
    * `numericAttrBinning (bool)` - If `True` - numeric attributes are split using binning, if `False` - binary split is performed.
    * `dataRange ((float, float) (default=None))` - Tuple corresponding to the range of values of the current node produced after binning.
    * `maxNumRecordsToSkipSplitting (int)` - Directly return a leaf node if number of records is lower than this value. Prevents overfitting.
    * `minNumRecordsLeafNode (int) (default=10)` - Minimum number of records in a leaf node. A split is ignored if there is a subset with lower number of elements than this value.
    * `repeatAttributes (bool) (default=False)` - If `True` - multiple splits can be performed over the same attribute, if `False` - once an attribute is part of a split it is no longer considered in further splits.

    ##### Returns
    * `node (Node)` - Leaf node if stopping criterium is met, otherwise - decision node.

* `generateTwigsDict(node)` - Traverses a tree and fills in `twigs (dict(Node))` class property with all current twigs (nodes whose children are all leaf nodes).

    ##### Parameters
    * `node (Node) (default=None)` - If `None` - clears the dictionary and starts from `treeRoot` node.

    ##### Returns
    * `None`

* `prune(maxNumOfLeafNodes)` - Prunes out portions of the tree that result in the least information gain. Continues while number of leaf nodes is more than desired.

    ##### Parameters
    * `maxNumOfLeafNodes (int)` - Pruning stops if number of leaf nodes becomes less than or equal to this value. 

    ##### Returns
    * `None`

* `traverseTree(row, node)` - Recursively traverses a structure of Nodes using values in dataframe row for navigation. Returns the class corresponding to the leaf node that was reached.

    ##### Parameters
    * `row (pandas.Series)` - Row from dataframe to use. 
    * `node (Node) (default=None)` - - If `None` - starts from `treeRoot` node. 

    ##### Returns
    * `node.name (str)` - The class corresponding to the leaf node that was reached.

* `classify(data)` - Classifies records in given dataset using already constructed tree.

    ##### Parameters
    * `data (pandas.DataFrame)` - Dataframe to use.

    ##### Returns
    * `None`     
    
### Node(name, threshold, isLeafNode, data, dataRange, gain)

* Class containing information for a single tree node.

#### Parameters

* `name (str)` - Name of splitting attibute for decision nodes, class (value of target variable) for leaf nodes.
* `threshold (float)` - Used only if binary split of numeric attribute is performed.
* `isLeafNode (bool)` - `True` - leaf node, `False` - decision node. 
* `data (pandas.DataFrame)` - Subset of training data.
* `dataRange ((float, float)) (default=None)` - Used only if binning of numeric attribute is performed.
* `gain (float) (default=None)` - Assigned only to decision nodes. Information gain of the node. Used for pruning.

#### Properties

* `name (str)` - Name of the node.
* `threshold (float)` - Number indicating the split point of a numeric binary split in the node.
* `isLeafNode (bool)` - Signifies whether the node is a leaf.
* `data (pandas.DataFrame)` - Records from the initial data that are part of the node.
* `dataRange ((float, float))` - Tuple corresponding to the range of values for data in the node.
* `gain (float)` - Information gain of the node. Used for pruning.
* `childNodes (list(Node))` - List of child nodes resulting from node splitting.

#### Methods
* `print(indentation, numOfTabsBetweenLevels, file)` - Traverses all child nodes of current node and recursively prints info about them in a tree-like fashion. Printing pattern depends on type of attribute (categorical or numerical) and type of split (binary or using binning).

    ##### Parameters
    * `indentation (str) (default='')` - Initial printing indentation.
    * `numOfTabsBetweenLevels (int) (default=1)` - Increasing this number increases the distance between tree levels when printing. If value is less than or equal to 0 no indentation between levels is used.
    * `file (file object) (default=None)` - File object to print the tree to. If `None` - standart output is used.

    ##### Returns
    * `None`

### DataPrep(data, attributes)

* Class which methods help with data preparation

#### Parameters

* `data (pandas.DataFrame)` - Subset of training data.
* `attributes (list)` - List of attribute names considered for splitting on.

#### Methods

* `fixCapitalization(self)` - Lowers the cases of all attributes.
* `checkForNumericOutliersWithBoxPlot(self)` - Creates a box plot which helps you to determine if there are any numeric outliers.
* `checkForNumericOutliersWithDescription(self)` - Creates desctiptions of all numerical attributes which helps you to determine if there are any numberic outliers.
* `checkForCategoricalOutliers(self)` - Creates a plot, which helps you to determine if there are any categorical outliers.
* `mergeCategoricalOutliers(self, attrName, thresholdPercent)` - Merges categorical outliers, classifies them as 'Other'.
	
	##### Parameters
	* `attrName (str)` - The name of the attribute with the outliers that you want to merge.
	* `thresholdPercent` - The records which are encountered under that percentage will be merged.

* `createMissingDataHeapMap(self)` - Creates a heap map which helps you to determine if there is any missing data.
* `checkMissingDataPercentageList(self)` - Creates a list which helps you to determine if there is any missing data.
* `checkForNonUniqueValue(self)` - Checks for attributes with only non-unique values. If any - the attribute is removed.
* `checkForDuplicatesAndRemoveIfAny(self)` - Checks for duplicates and removes them if any.

### DecisionMaker(data, treeRoot)

* Second variation of a classifier, which classifies records in given dataset using already constructed tree.

#### Parameters

* `data (pandas.DataFrame)` - Subset of training data.
* `treeRoot (str)` - The name of the root of the tree.

#### Properties

* `new_dataSet (pandas.DataFrame)` - A new dataset, where the classified records will be saved.
* `last_column_name (str)` - The name of the last attrbitue(the attribute that we expected to be classified by our algorithm).

#### Methods

* `DecisionMaking(node)` - The method, which makes the classification.
	
	#### Parameters
	* `node (Node)` - You should always pass the treeRoot node as node in this method, because the crawling of the tree always begins with the root of the tree.
	
	#### Properties
	* `branches_names (list)` - List which contains the names of all the branches of a node.
	* `isCategorical (bool)` - A property which helps us to determine whether an attribute is categorical or numeric.
	* `node_column_index (int)` - A property which holds the index of the node column

The method first checks whether the root of the tree is NOT signed/unsigned integer; floating point; complex floating point, i.e. if the root is categorical. If it is
categorical the method adds the branches names of the root in branches_names. For categorical nodes the branches_names are simply the records. For example if we have node
'Assets' the branches names will be 'High', 'Low', 'Medium'. If the root is not categorical the method again adds all the branches names in the list branches_names, but the
names for numeric nodes are not as simple as the names of categorical nodes. The names of the numeric nodes can be intervals, like 'Age ∈ (20.25, 22.5)' or just one or two
values, for example 'Age [23]' or 'Age [23, 33]'. All the branches names are added in such way in the list so that we can use them as if statements. For example the 
branch_name 'Age ∈ (20.25, 22.5)' in the three will be added in the list as '22 > 20.25 and 22 < 22.5', where '22' is the actual 'Age' of the current node. Then after we
have the branches names of the current node in the list out method starts itterating through the rows of the dataset. If the current node is categorical we are checking
whether the current row value branch is leaf. If it is leaf, we are adding the value of the leaf to the last column on the row (we are classificating the record) and we are 
calling out method again with the treeRoot for the next row.If it is not leaf we are recursively calling the method with the non leaf node. If the current node is numeric our 
method starts itterating through branches names. For each branch name our method replaces node name in branch name with row value, so that we have 'ROW_VALUE > 20.25 and
ROW_VALUE < 22.5' instead of 'Age > 20.25 and Age < 22.5' for example. The next step in our method is to use the branch name as an if statement. If the if statement passes,
i.e. if our ROW_VALUE is bigger than 20.25 and smaller than 22.5 for example, we are checking whether the child that the branch led us to is a leaf. If it is leaf, we are
adding the value of the leaf to the last column on the row (we are classificating the record) and we are calling out method again with the treeRoot for the next row. If it is
not leaf we are recursively calling the method with the non leaf node.
