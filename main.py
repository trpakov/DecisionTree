import pandas as pd
import DecisionTree as dt

# print((data.iloc[:,-1].to_numpy() == "Iris-setosa"))
# print(     [ (x, np.count_nonzero(data.iloc[:,-1].to_numpy() == x)) for x in set(data.iloc[:,-1])  ]     )
# print(len(data))

data = pd.read_csv("data/iris.csv")
dtg = dt.DecisionTreeGenerator(data)
print(dtg.getNumberOfRecordsInEachClass(data))
print(dtg.calculateEntopy(data))
print(dtg.getClassWithMostRecords(data))
print(dtg.getNumberOfRecordsInClass(data, "Iris-virginic"))
print(dtg.calculateMeasureOfGoodness(data, [data.head(100), data.tail(50)]))
