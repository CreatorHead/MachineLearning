from math import log
import operator


def calcShannonEnt(dataSet):
    """
    The calcShannonEnt(dataSet) is straightforward, first you calculate a count of the number of instances in the dataset.
    This could have been calculated inline, but it's used multiple times in the code, so an explicit
    variable is created for it. Next, you create a dictionary whose keys are the values in the final
    column.
    1.  If a key was not encountered previously, one is created. For each key, you keep track of how many
        times this label occurs. Finally, you use the frequency of all the different labels to calculate
        the probability of that label. This probability is used to calculate the Shanon entropy.
    2.  And, you sum this up for all the labels.
    :return: float
    """
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


def createDataSet():
    """
    Simple data about fish identification
    :return: dataSet, labels
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    It takes three inputs: the dataset we'll split, the feature we'll split on, and the value of the feature to
    return. Created a new list each time because you'll be calling this function multiple times on the same dataset
    and you don't want the original dataset modified. Our dataset is a list of list; you iterate over every item in
    the list and if it contains the value you're looking for, you'll add it to your newly created list. Inside the if
    statement, you cut out the feature that you split on. Think of it this way: once you've split on a feature,
    you're finished with that feature. Here, we're using the extrend() and append() methods of the Python list type.

    :param dataSet:
    :param axis:
    :param value:
    :return: list
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


'''Now we'll combine the Shannon entropy calculation and the splitDataSet function to cycle through the dataset and 
   decide which feature is the best to split on. Using the entropy calculation tells you which split best organized your 
   data. '''


def chooseBestFeatureToSplit(dataSet):

    """
    We made few assumption about the data. The first assumption about the data is that it comes in the form of a list
    of lists, and all these list are of equal size. The next assumption is that the last column in the data or the
    last item in each instance is the class label of that instance. You use these assumption in the first line of the
    function to find out how many features you have available in the given dataset. We didn't make any assumption on
    the type of data in the lists. it could be a number or a string; it doesn't matter. The next part of the code
    calculates the Shannon entropy of the whole dataset before any splitting has occurred. This gives you the base
    disorder, which you'll later compare to the post split disorder measurements. The first for loop loops over all the
    features in our dataset, or all the possible values present in the data. Next, you use the python native set data
    type. Creating a new set from a list is one of the fastest ways of getting the unique values out of list in Python.
    Now, you go through the unique values of this feature and split the data for each feature. The new entropy is
    calculated and summed up for all the unique values of that features. The information gain is the reduction in entropy
    or the reduction in messiness.Finally you compare the information gain among all the features and return the index of
    the best feature to split on.

    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1  # the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1

    # Create unique list of class labels
    for i in range(numFeatures):  # iterate over all the features
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature
        uniqueVals = set(featList)  # get a set of unique values
        newEntropy = 0.0

        # Calculate entropy for each split
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy  # calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):  # compare this to the best gain so far
            bestInfoGain = infoGain  # if better than current best, set to best
            bestFeature = i
    return bestFeature  # returns an integer


'''
If our dataset has run out of attributes but the class labels are not all the same, you must decide what to call that 
leaf node. In this situation, you'll take a majority vote.
'''


def majorityCnt(classList):
    """
    This function takes a list of class names and then creates a dictionary whose keys are the unique values in
    classList, and the object of the dictionary is thr frequency of occurrence of each class label from classList.
    Finally, you use the operator to sort the dictionary by the keys and return the class that occurs with the
    greatest frequency.
    :param classList:
    :return: number
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    The list of lables contains a label for each of the features in the dataset. The algorithm could function without
    this, but it would be difficult to make any sense of the data. You first create a list of all the class labels in
    our dataset and call this classList. The first stopping condition is that if all the class labels are the same,
    then you return this label. The second stopping condition is the case when there are no more features to split.
    If you don't meet the stopping conditions, then you use the function "chooseBestFeatureToSplit" to choose the best
    feature. We'll use the python dictionary to store the tree. You could have created a special data type, but it's
    not necessary. The myTree dictionary will be used to store the tree. We'll get all the unique values from the
    dataset for our chosen feature: bestFeat. The unique value code uses sets. Finally, we iterate over all the unique
    values from our chosen feature and recursively call createTree() for each split of the dataset. This value is
    inserted into our myTree dictionary, so you end up with a lot of nested dictionaries representing our tree. Note
    that subLabels = labels[:] line make a copy of labels and places it in a new list called subLabels. We do this
    because Python passes list by reference and you'd like the original list to be the same every time you call
    createTree().
    :param dataSet:
    :param labels:
    :return: dictionary
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


myDataSet, labels = createDataSet()
featureIndex = chooseBestFeatureToSplit(myDataSet)
print(featureIndex)
