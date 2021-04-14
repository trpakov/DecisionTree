import numpy as np
import pandas as pd
import DecisionTree as dt
import DataPreparation as dp

data1 = pd.read_csv("data/iris.csv")
dtg1 = dt.DecisionTreeGenerator(data1)
#dtg1.generate()
#dtg1.treeRoot.print()

print()

data2 = pd.read_csv("data/creditRisk.csv")
dtg2 = dt.DecisionTreeGenerator(data2)
print(data2)

print()

data3 = pd.read_csv("data/flavors_of_cacao.csv")
data3.columns = ['company','bar_name','ref','review_year','cocoa_percent',
                'company_location','rating','bean_type','broad_bean_origin']
dpp = dp.DataPrep(data3, list(data3)[:-1])
data3 = dpp.mergeCategoricalOutliers('company_location', 1)
print(data3.company_location.value_counts())
