import numpy as np
import pandas as pd
import DecisionTree as dt

# print((data.iloc[:,-1].to_numpy() == "Iris-setosa"))
# print(     [ (x, np.count_nonzero(data.iloc[:,-1].to_numpy() == x)) for x in set(data.iloc[:,-1])  ]     )
# print(len(data))

data = pd.read_csv("data/iris.csv")
dtg = dt.DecisionTreeGenerator(data)
#dtg.checkForCategoricalData()
print(dtg.getNumberOfRecordsInEachClass(data))
print(dtg.calculateEntopy(data))
print(dtg.getClassWithMostRecords(data))
print(dtg.getNumberOfRecordsInClass(data, "Iris-virginic"))
print(dtg.calculateMeasureOfGoodness(data, [data.head(100), data.tail(50)]))

med = dtg.splitData(dtg.data, dtg.inputVars)
print(med)
med2 = dtg.splitData(med[1][1], [x for x in dtg.inputVars if x != 'petal_width'])
print(med2)
med21 = dtg.splitData(med2[1][0], [x for x in dtg.inputVars if x not in ['petal_width', 'petal_length' ]])
med22 = dtg.splitData(med2[1][1], [x for x in dtg.inputVars if x not in ['petal_width', 'petal_length' ]])
print(med21)
print(med22)

# print(data['Income($1000s)'].iloc[2:4])
# for i,d in enumerate(data['Income($1000s)']):
#     print(i, d)

