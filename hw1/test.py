import numpy

def readTestData(fileName):
    dataSet = []
    file = open(fileName, 'rb')
    firstLine = True
    for line in file.readlines():
        if firstLine:
            firstLine = False
            continue
        dataRow = line.strip().split(',')
        dataSet.append(map(float, dataRow[1:]))
    return dataSet

def test(dataSet, coefficient):
    lenOfCase = len(dataSet)
    dataSet = numpy.array(dataSet)
    result = dataSet.dot(coefficient)
    return result