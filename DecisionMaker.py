class DecisionMaker:

    def __init__(self, data):
        self.data = data

    counter = 0

    def DecisionMaking(self, node):
        branches_names = []

        #Getting node column index
        node_column_index = self.data.columns.get_loc(node.name)

        #Adding branches names in list
        for child in node.childNodes:
            branches_names.append(child.data.iloc[0, node_column_index])

        for row in self.data.iloc[self.counter:].iterrows():

            print(row[1][node.name])
            #Checking if row value branching is leaf
            if node.childNodes[branches_names.index(row[1][node.name])].isLeafNode:

                #If it is leaf we are adding the value of the leaf to the last column on the row
                row[1][-1] = node.childNodes[branches_names.index(row[1][node.name])].name
                self.counter += 1
                print(row[1])

            else:
                #If it is not leaf we are recursively calling the method with the non leaf node
                self.DecisionMaking(node.childNodes[branches_names.index(row[1][node.name])])

        return self.data
