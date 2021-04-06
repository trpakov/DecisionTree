import numpy as np
import pandas as pd
import DecisionTree as dt
import DataPreparation as dp

data1 = pd.read_csv("data/iris.csv")
dtg1 = dt.DecisionTreeGenerator(data1)
dtg1.generate()
#dtg1.treeRoot.print()

print()

data2 = pd.read_csv("data/creditRisk.csv")
dpp = dp.DataPrep(data2, list(data2)[:-1])
dtg2 = dt.DecisionTreeGenerator(data2)
dtg2.generate()
#dtg2.treeRoot.print()

