import numpy as np
from numpy.linalg import inv


def bHeadCal(x, y):
    xArray = np.array(x)
    yArray = np.array(y)
    antecedent = np.linalg.inv(np.dot(np.transpose(xArray), xArray))
    consequent = np.dot(np.transpose(xArray), yArray)
    return np.dot(antecedent, consequent)


def predict(parmeters, b):
    return np.dot(parmeters, b)


fileName = "data_class.txt"
height = []
weight = []
sex = []
with open(fileName, 'r', encoding="utf-8") as f:
    for line in f.readlines():
        data = list(line.split("\t"))
        height.append([float(data[1])])
        weight.append([float(data[1])])
        if data[2] == '男\n':
            sex.append([0])
        else:
            sex.append([1])

inputData = []
for i in range(len(height)):
    inputData.append(np.concatenate([height[i], sex[i]]))

bHead = bHeadCal(inputData, weight)
print(bHead)
print(predict([150, 1], bHead))