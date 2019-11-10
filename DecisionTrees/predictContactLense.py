import pathlib

from DecisionTrees.treePlotter import createPlot
from DecisionTrees.trees import createTree

filePath = pathlib.Path(__file__).parent / '../resources/lenses.txt'
fr = open(filePath)
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = createTree(lenses, lensesLabels)
print(lensesTree)
createPlot(lensesTree)

