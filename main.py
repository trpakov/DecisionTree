import numpy as np
import pandas as pd
import DecisionTree as dt

data1 = pd.read_csv("data/iris.csv")
dtg1 = dt.DecisionTreeGenerator(data1)
dtg1.generate()
dtg1.treeRoot.print()

print()

data2 = pd.read_csv("data/creditRisk.csv")
dtg2 = dt.DecisionTreeGenerator(data2)
dtg2.generate()
dtg2.treeRoot.print()