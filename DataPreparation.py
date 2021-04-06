import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataPrep:

    def __init__(self, data, attributes):
        self.data = data
        self.attributes = attributes

    def fixCapitalization(self):
        for attr in self.attributes:
            if str(self.data[attr].dtype) == 'object' or str(self.data[attr].dtype) == 'category':
                self.data[attr] = self.data[attr].str.lower()

    def checkForNumericOutliersWithBoxPlot(self):
        for attr in self.attributes:
            if str(self.data[attr].dtype) != 'object' and str(self.data[attr].dtype) != 'category':
                self.data.boxplot(column=[attr])
                plt.show()

    def checkForNumericOutliersWithDescription(self):
        for attr in self.attributes:
            if str(self.data[attr].dtype) != 'object' and str(self.data[attr].dtype) != 'category':
                print(self.data[attr].describe())
                print()

    def checkForCategoricalOutliers(self):
        for attr in self.attributes:
            if str(self.data[attr].dtype) == 'object' or str(self.data[attr].dtype) == 'category':
                self.data[attr].value_counts().plot.bar()
                plt.show()
                print()

    def createMissingDataHeapMap(self):
        '''For smaller number of features'''
        cols = self.data.columns[:]
        colours = ['r', 'g']  # specify the colours - yellow is missing. blue is not missing.
        sns.heatmap(self.data[cols].isnull(), cmap=sns.color_palette(colours))
        plt.show()

    def checkMissingDataPercentageList(self):
        '''For bigger number of features'''
        for col in self.data.columns:
            pct_missing = np.mean(self.data[col].isnull())
            print('{} - {}%'.format(col, round(pct_missing * 100)))

    def checkForNonUniqueValue(self):
        condition = False

        for col in self.data.columns:
            if len(self.data[col].unique()) < 2:
                condition = True
                self.data.drop(col, axis=1, inplace=True)
        if condition is False:
            print("No columns with only 1 value found. You can proceed further. :)")

    def checkForDuplicatesAndRemoveIfAny(self):
        if self.data.duplicated().any() is True:
            print("There are duplicates in your DataFrame \n" + self.data[self.data.duplicated])
            self.data.drop_duplicates(inplace=True)
        else:
            print("There are no duplicates in your DataFrame.")
