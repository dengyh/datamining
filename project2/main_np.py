import math
import time
import random
import numpy

from collections import Counter

class TreeNode:
    def __init__(self, attribute, critical):
        self.attribute = attribute
        self.critical = critical
        self.leftChild = None
        self.rightChild = None

class Classification:
    def __init__(self, value):
        self.value = value

def getSamples():
    file = open('train.csv')
    samples = []
    firstLine = file.readline()
    for line in file.readlines():
        samples.append(map(float, line.strip().split(', ')[1:]))
    attributes = set(range(len(samples[0] ) - 1))
    file.close()
    return numpy.array(samples), attributes

def getTests():
    file = open('test.csv')
    tests = []
    firstLine = file.readline()
    for line in file.readlines():
        tests.append(map(float, line.strip().split(', ')[1:]))
    file.close()
    return numpy.array(tests)

def buildDecisionTree(samples, attributes, parenSamples):
    if len(samples) == 0:
        return getPluralityClassification(parenSamples)
    if hasSameClassification(samples):
        return Classification(samples[0, -1])
    if len(attributes) == 0:
        return getPluralityClassification(parenSamples)
    attributeValue, attributeCritical, attributeEntropy= getMaxImportant(attributes, samples)
    tree = TreeNode(attributeValue, attributeCritical)
    attributes.remove(attributeValue)
    leftSamples, rightSamples = splitSamples(samples, attributeValue, tree.critical)
    tree.leftChild = buildDecisionTree(leftSamples, attributes, samples)
    tree.rightChild = buildDecisionTree(rightSamples, attributes, samples)
    attributes.append(attributeValue)
    return tree

def getPluralityClassification(samples):
    counter = Counter(samples[:,-1])
    return Classification(counter.most_common(1)[0][0])

def hasSameClassification(samples):
    tempSet = numpy.unique(samples[:,-1])
    return len(tempSet) == 1

def getMaxImportant(attributes, samples):
    attributeValue, attributeCritical, attributeEntropy = '', 0, -1
    for attr in attributes:
        entropy, critical = getImportantValue(samples, attr)
        if entropy > attributeEntropy:
            attributeValue = attr
            attributeEntropy = entropy
            attributeCritical = critical
    return attributeValue, attributeCritical, attributeEntropy

def getImportantValue(samples, attribute):
    criticals = list(getCriticalValues(samples, attribute))
    results = list(getRemainderValues(samples, attribute, criticals))
    return max(results, key=lambda x: x[0])

def getCriticalValues(samples, attribute):
    average = numpy.average(samples[:,attribute])
    yield average

def getRemainderValues(samples, attribute, criticals):
    totalLength = len(samples)
    resultCounter = Counter(samples[:,-1])
    samplesEntropy = getEntropyFromCounter(resultCounter, totalLength)
    for critical in criticals:
        totalLarger, totalSmaller = 0, 0
        largerCounter, smallerCounter = Counter(), Counter()
        for item in samples:
            if item[attribute] > critical:
                totalLarger += 1
                largerCounter[item[-1]] += 1
            else:
                totalSmaller += 1
                smallerCounter[item[-1]] += 1
        remainder = 0
        if notEqual(totalSmaller, 0.0):
            remainder += totalSmaller * 1.0 / totalLength * getEntropyFromCounter(smallerCounter, totalSmaller)
        if notEqual(totalLarger, 0.0):
            remainder += totalLarger * 1.0 / totalLength * getEntropyFromCounter(largerCounter, totalLarger)
        yield samplesEntropy - remainder, critical

def getEntropyFromCounter(counter, length):
    entropy = 0
    for value in counter.values():
        entropy -= getEntropy(float(value) / length)
    return entropy

def getEntropy(value):
    if notEqual(value, 1.0) and notEqual(value, 0.0):
        return value * log2(value)
    return 0.0

def notEqual(valueA, valueB):
    return not (abs(valueA - valueB) < 0.0001)

def log2(value):
    return math.log(value) / math.log(2)

def splitSamples(samples, attribute, critical):
    leftSamples = numpy.array(filter(lambda x: x[attribute] < critical, samples))
    rightSamples = numpy.array(filter(lambda x: x[attribute] >= critical, samples))
    return leftSamples, rightSamples

def findPath(tree, sample):
    if isinstance(tree, Classification):
        return tree.value
    if sample[tree.attribute] < tree.critical:
        return findPath(tree.leftChild, sample)
    else:
        return findPath(tree.rightChild, sample)

def getRandomAttributes(attributes):
    attributes = list(attributes)
    length = len(attributes)
    targetLength = int(math.sqrt(length))
    tempAttributes = set()
    while len(tempAttributes) != targetLength:
        index = random.randint(0, length - 1)
        tempAttributes.add(attributes[index])
    return list(tempAttributes)

if __name__ == '__main__':
    start = time.time()
    samples, attributes = getSamples()
    tests = getTests()
    for index in xrange(100):
        tempAttributes = getRandomAttributes(attributes)
        tree = buildDecisionTree(samples, tempAttributes, samples)
        file = open('result/result' + str(index) + '.csv', 'w')
        file.write('id,label\n')
        for i, item in enumerate(tests):
            value = findPath(tree, item)
            file.write('%d,%s\n' % (i, int(value)))
        file.close()
        print 'Complete %d tree' % index
    end = time.time()
    print end - start