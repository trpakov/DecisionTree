class DecisionMaker:

    def __init__(self, data, treeRoot):
        self.data = data
        self.treeRoot = treeRoot
        self.new_dataSet = self.data
        self.last_column_name = list(self.data.columns)[-1]

    counter = 0
    counter2 = 0

    def DecisionMaking(self, node):
        branches_names = []
        new_dataSet = self.data
        isCategorical = False

        #Getting node column index
        node_column_index = self.data.columns.get_loc(node.name)

        #Checking if the column is not signed/unsigned integer; floating point; complex floating point
        if not self.data[node.name].dtype.kind in 'iufc':
            isCategorical = True
            # Adding branches names in list
            for child in node.childNodes:
                branches_names.append(child.data.iloc[0, node_column_index])

        # Adding branches names in list
        else:
            isCategorical = False
            for index, child in enumerate(node.childNodes):
                if node.threshold is None:
                    if child.dataRange is None:
                        branches_names.append(str(node.name) + ' ' + str(child.data.loc[:, node.name].unique().tolist()))
                    else:
                        branch_name = str(child.dataRange).replace('(', '').replace(',', '').replace(')', '').split(' ')
                        branches_names.append(node.name + ' > ' + branch_name[0] + ' and ' + node.name + ' < ' + branch_name[1])
                else:
                    branches_names.append([node.name + ' < ' + str(node.threshold) + ' or ' + node.name + ' == ', node.name + ' > '][index] + str(node.threshold))

        for row in self.data.iloc[self.counter:].iterrows():

            if self.counter == len(self.data.index):
                break

            if isCategorical:
                #Checking if row value branching is leaf
                if node.childNodes[branches_names.index(row[1][node.name])].isLeafNode:

                    #If it is leaf we are adding the value of the leaf to the last column on the row
                    row[1][-1] = node.childNodes[branches_names.index(row[1][node.name])].name
                    new_dataSet.at[self.counter, 'Credit_Risk'] = node.childNodes[branches_names.index(row[1][node.name])].name
                    self.counter += 1
                    self.DecisionMaking(self.treeRoot)
                    return

                else:
                    #If it is not leaf we are recursively calling the method with the non leaf node
                    self.DecisionMaking(node.childNodes[branches_names.index(row[1][node.name])])
            else:

                self.counter2 = 0

                for branch_name in branches_names:

                    #Replacing node name in branch name with row value
                    branch_name = branch_name.replace(node.name, str(row[1][node.name]))

                    print(branch_name)
                    if eval(branch_name):

                        print("in")
                        if node.childNodes[self.counter2].isLeafNode:
                            row[1][-1] = node.childNodes[self.counter2].name
                            new_dataSet.at[self.counter, 'species'] = node.childNodes[self.counter2].name
                            self.counter += 1
                            self.DecisionMaking(self.treeRoot)
                            return

                        else:
                            self.DecisionMaking(node.childNodes[self.counter2])

                    self.counter2 += 1

        self.data = new_dataSet
