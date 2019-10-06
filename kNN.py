from numpy import *
import operator
import kNN

"""
Python also has the ability to return multiple values from a function call ,
something missing from many other languages. In this case the return values should
be a comma-separated list of values and Python then constructs a tuple and returns
this to the caller
"""
def createDataSet():
    group = array([[1.0, 1.1],
                   [1.0, 1.0],
                   [0, 0],
                   [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
