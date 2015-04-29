import numpy

def readTrainData(fileName):
    dataSet = []
    resultSet = []
    file = open(fileName, 'rb')
    firstLine = True
    for line in file.readlines():
        if firstLine:
            firstLine = False
            continue
        dataRow = line.strip().split(',')
        dataSet.append(map(float, dataRow[1:-1]))
        resultSet.append(float(dataRow[-1]))
    return dataSet, resultSet

def train(dataSet, resultSet, learnRate, learnTime):
    lenOfCase = len(dataSet)
    lenOfAttr = len(dataSet[0])
    dataSet = numpy.array(dataSet)
    resultSet = numpy.array(resultSet)
    coefficient = numpy.random.rand(lenOfAttr)
    for time in xrange(learnTime):
        print 'Training %d times...' % time
        diff = (dataSet.dot(coefficient) - resultSet).dot(dataSet) / lenOfCase * learnRate
        coefficient -= diff
    return coefficient
