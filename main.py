import numpy as np
import pandas as pd
import DecisionTree as dt
import DataPreparation as dp
import DecisionMaker as dm

#data1 = pd.read_csv("data/iris.csv")
#dtg1 = dt.DecisionTreeGenerator(data1)
#dtg1.generate()
#dtg1.treeRoot.print()

print()

data2 = pd.read_csv("data/iris.csv")
#dpp = dp.DataPrep(data2, list(data2)[:-1])
dtg2 = dt.DecisionTreeGenerator(data2)
dtg2.generate(maxNumRecordsToSkipSplitting=1)
dtg2.treeRoot.print()
data3 = pd.read_csv("data/iris2.csv")
decisionMaker = dm.DecisionMaker(data3, dtg2.treeRoot)
decisionMaker.DecisionMaking(dtg2.treeRoot)
print(data3)

#data2 = pd.read_csv("data/iris.csv")
#dtg2 = dt.DecisionTreeGenerator(data2)
#dtg2.generate()
#dtg2.treeRoot.print()
# dtg2.treeRoot.print()

#data3 = pd.read_csv("data/titanic.csv")
#data3 = data3.astype({'Pclass':'category', 'Siblings/Spouses Aboard':'category', 'Parents/Children Aboard':'category'}, copy=False)
#dtg3 = dt.DecisionTreeGenerator(data3)
#dtg3.generate(maxNumRecordsToSkipSplitting=1)
#dtg3.treeRoot.print()
#data3 = pd.read_csv("data/titanic2.csv")
#decisionMaker = dm.DecisionMaker(data3, dtg3.treeRoot)
#decisionMaker.DecisionMaking(dtg3.treeRoot)
#print(dtg3.treeRoot.childNodes[0].name)

#dtg3.prune(10)
#f = open('data/titanic.txt', mode='w', encoding='utf-8')
#dtg3.treeRoot.print(file=f)

# testData = pd.read_csv("data/titanicTest.csv")
# print(dtg3.classify(testData.iloc[:,:-1]))

# x = data3["Age"].values
# y = data3["Survived"].values
# optb = MulticlassOptimalBinning(min_n_bins=2, max_n_bins=4)
# optb.fit(x, y)
# bt = optb.binning_table.build()
# print(bt)
# btRanges = bt['Bin'].head(-3)
# btRanges.iloc[0] = btRanges.iloc[0].replace('-inf', "'-inf'")
# btRanges.iloc[-1] = btRanges.iloc[-1].replace('inf', "'inf'")
# lst = [literal_eval(x.replace('[', '(')) for x in btRanges]
# lst = [(-np.inf, x[1]) if x[0] == '-inf' else ((x[0], np.inf) if x[1] == 'inf' else (x[0], x[1])) for x in lst]

# x = data2["Savings"]
# y = data2["Credit_Risk"].astype('category').cat.codes.values
# # y = set(y)
# # y = [x for x in range(len(y))]
# print(y)
# optb = OptimalBinning(dtype='categorical', min_n_bins=2, max_n_bins=5)
# optb.fit(x, y)
# bt = optb.binning_table.build()
# print(bt)


#print()

#data3 = pd.read_csv("data/flavors_of_cacao.csv")
#data3.columns = ['company','bar_name','ref','review_year','cocoa_percent',
#                'company_location','rating','bean_type','broad_bean_origin']
#dpp = dp.DataPrep(data3, list(data3)[:-1])
#data3 = dpp.mergeCategoricalOutliers('company_location', 1)
#print(data3.company_location.value_counts())
