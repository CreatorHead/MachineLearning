from codes.kNN.kNN import *

""" First Test Sample """
# group, labels = create_dataSet()
# print(classify0([0, 0], group, labels, 3))

""" 2nd Test Sample: Dating ML Model """
# draw_time_spent_in_video_game_VS_Ice_cream_consumed()
# draw_Flyier_Mies_VS_time_spent_in_video_game()


# filePath = pathlib.Path(__file__).parent / '../../resources/datingTestSet.txt'
# datingDataMat, datingLabels = file2matrix(filePath)
# normMat, ranges, minVals = autoNorm(datingDataMat)
# print(normMat)

# datingClassTest()

# classifyPerson()

# filePath = pathlib.Path(__file__).parent / '../../resources/testDigits/0_13.txt'
# testVector = img2Vector(filePath)
# print(testVector[0, 0:31])

handwritingClassTest()
