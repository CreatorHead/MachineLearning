from math import log


'''
The code is straightforward, first you calculate a count of the number of instances in the dataset.
This could have been calculated inline, but it's used multiple times in the code, so an explicit 
variable is created for it. Next, you create a dictionary whose keys are the values in the final 
column. 

'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:  # the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)  # log base 2
    return shannonEnt
