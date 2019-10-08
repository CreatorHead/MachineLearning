from numpy import *

# creates random array of size 4 * 4
randArr = random.rand(4, 4)

# convert an array to a matrix
randMat = mat(randArr)

arr =[[2,1,1],[3,2,1],[2,1,2]]
arrMat = mat(arr)
inverseArrMat = arrMat.I
print(inverseArrMat)
print(arrMat * inverseArrMat) # should give an identity matrix
print(eye(4))
