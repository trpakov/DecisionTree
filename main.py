import numpy as np
import pandas as pd
import DecisionTree as dt
import DataPreparation as dp
import DecisionMaker as dm

#data1 = pd.read_csv("data/iris.csv")
#dtg1 = dt.DecisionTreeGenerator(data1)
#dtg1.generate()
#dtg1.treeRoot.print()

# print()
# data2 = pd.read_csv("data/titanic2.csv")
# data2 = data2.astype({'Pclass':'category', 'Siblings/Spouses Aboard':'category', 'Parents/Children Aboard':'category'}, copy=False)
# dpp = dp.DataPrep(data2, list(data2)[:-1])
# dtg2 = dt.DecisionTreeGenerator(data2)
# dtg2.generate()
# dtg2.treeRoot.print()
# data3 = pd.read_csv("data/titanic3.csv")
# decisionMaker = dm.DecisionMaker(data3, dtg2.treeRoot)
# decisionMaker.DecisionMaking(dtg2.treeRoot)
# print(data3)

#data2 = pd.read_csv("data/iris.csv")
#dtg2 = dt.DecisionTreeGenerator(data2)
#dtg2.generate()
#dtg2.treeRoot.print()
# dtg2.treeRoot.print()

# data3 = pd.read_csv("data/creditRisk2.csv")
# decisionMaker = dm.DecisionMaker(data3, dtg2.treeRoot)
# decisionMaker.DecisionMaking(dtg2.treeRoot)
# print(data3)
# dtg2.treeRoot.print()

data3 = pd.read_csv("data/titanic2.csv")
data3 = data3.astype({'Pclass':'category', 'Siblings/Spouses Aboard':'category', 'Parents/Children Aboard':'category'}, copy=False)
dtg3 = dt.DecisionTreeGenerator(data3)
dtg3.generate()

# decisionMaker = dm.DecisionMaker(data3, dtg3.treeRoot)
# decisionMaker.DecisionMaking(dtg3.treeRoot)
# print(dtg3.treeRoot.childNodes[0].name)

#dtg3.prune(10)
f = open('data/titanic.txt', mode='w', encoding='utf-8')
dtg3.treeRoot.print(file=f)
f.close()

testData = pd.read_csv("data/TitanicTest.csv")
dtg3.classify(testData)
testData.to_csv('data/TitanicResult.csv', index=False)

#print()

#data3 = pd.read_csv("data/flavors_of_cacao.csv")
#data3.columns = ['company','bar_name','ref','review_year','cocoa_percent',
#                'company_location','rating','bean_type','broad_bean_origin']
#dpp = dp.DataPrep(data3, list(data3)[:-1])
#data3 = dpp.mergeCategoricalOutliers('company_location', 1)
#print(data3.company_location.value_counts())
