import pathlib
from numpy import *
import matplotlib.pyplot as plt
from codes.kNN.kNN import file2matrix


def draw_time_spent_in_video_game_VS_Ice_cream_consumed():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filePath = pathlib.Path(__file__).parent / '../../resources/datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filePath)
    # ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2])
    ax.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    ax.axis([-2, 25, -0.2, 2.0])
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Per Week')
    plt.show()


def draw_Flyier_Mies_VS_time_spent_in_video_game():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    filePath = pathlib.Path(__file__).parent / '../../resources/datingTestSet.txt'
    datingDataMat, datingLabels = file2matrix(filePath)
    # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    ax.axis([-2, 100000, -0.2, 25])
    plt.xlabel('Frequent Flyier Miles Earned Per Year')
    plt.ylabel('Percentage of Time spent playing video games')
    plt.show()
