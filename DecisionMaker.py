class DecisionMaker:

    def __init__(self, data, treeRoot):
        self.data = data
        self.treeRoot = treeRoot
        self.new_dataSet = self.data
        self.last_column_name = list(self.data.columns)[-1]

    counter = 0

    def DecisionMaking(self, node):
        branches_names = []
        new_dataSet = self.data

        #Getting node column index
        node_column_index = self.data.columns.get_loc(node.name)

        #Adding branches names in list
        for child in node.childNodes:
            branches_names.append(child.data.iloc[0, node_column_index])

        for row in self.data.iloc[self.counter:].iterrows():

            if self.counter == len(self.data.index):
                break

            #Checking if row value branching is leaf
            if node.childNodes[branches_names.index(row[1][node.name])].isLeafNode:

                #If it is leaf we are adding the value of the leaf to the last column on the row
                row[1][-1] = node.childNodes[branches_names.index(row[1][node.name])].name
                new_dataSet.at[self.counter, 'Credit_Risk'] = node.childNodes[branches_names.index(row[1][node.name])].name
                self.counter += 1
                print(row[1])
                self.DecisionMaking(self.treeRoot)

            else:
                #If it is not leaf we are recursively calling the method with the non leaf node
                self.DecisionMaking(node.childNodes[branches_names.index(row[1][node.name])])

        self.data = new_dataSet
