from test import readTestData, test
from train import readTrainData, train

def outputResult(result, fileName):
    file = open(fileName, 'w')
    lenOfResult = len(result)
    file.write('Id,reference\n')
    for index in xrange(lenOfResult):
        file.write('%d,%.6f\n' % (index, result[index]))
    file.close()

if __name__ == '__main__':
    trainDataSet, trainResultSet = readTrainData('data/train_temp.csv')
    coefficient = train(trainDataSet, trainResultSet, 0.01, 10000)
    testDataSet = readTestData('data/test_temp.csv')
    result = test(testDataSet, coefficient)
    outputResult(result, 'data/result.csv')
