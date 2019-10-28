from DecisionTrees.calcShanonEntropy import calcShannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


myDataSet, labels = createDataSet()
entropy = calcShannonEnt(myDataSet)
print(entropy)
