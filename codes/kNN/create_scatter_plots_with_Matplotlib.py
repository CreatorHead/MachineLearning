import pathlib

import matplotlib
import matplotlib.pyplot as plt
from codes.kNN.kNN import *

filePath = pathlib.Path(__file__).parent / '../../resources/datingTestSet.txt'
datingDataMat, datingLabels = file2matrix(filePath)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
plt.show()
