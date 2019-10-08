import pathlib

from codes.kNN.kNN import *

""" First Test Sample """
#group, labels = create_dataSet()
#print(classify0([0, 0], group, labels, 3))

""" 2nd Test Sample: Dating ML Model """
filePath = pathlib.Path(__file__).parent/'../../resources/datingTestSet.txt'
datingDataMat, datingLabels = file2matrix(filePath)

