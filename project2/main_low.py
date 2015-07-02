#!/usr/bin/python
import math
import time
import random

from collections import Counter

DEBUG = True
FILE = open('gain.txt', 'w')

class Sample:
    def __init__(self, input, attributes):
        attrs = input.strip().split(', ')
        values = attrs[1:-1]
        self.result = attrs[-1]
        self.values = {}
        for index, item in enumerate(attributes):
            self.values[item] = float(values[index])

class Test:
    def __init__(self, input, attributes):
        attrs = input.strip().split(', ')
        values = attrs[1:]
        self.values = {}
        for index, item in enumerate(attributes):
            self.values[item] = float(values[index])

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
    attributes = getAttributesFromFile(file.readline())
    for line in file.readlines():
        samples.append(Sample(line, attributes))
    file.close()
    return samples, attributes

def getTests():
    file = open('test.csv')
    samples = []
    firstLine = file.readline()
    attributes = firstLine.strip().split(',')[1:]
    for line in file.readlines():
        samples.append(Test(line, attributes))
    file.close()
    return samples

def getAttributesFromFile(line):
    return line.strip().split(',')[1:-1]

def buildDecisionTree(samples, attributes, parenSamples):
    if len(samples) == 0:
        return getPluralityClassification(parenSamples)
    if hasSameClassification(samples):
        return Classification(samples[0].result)
    if len(attributes) == 0:
        return getPluralityClassification(parenSamples)
    attributeValue, attributeCritical, attributeEntropy= getMaxImportant(attributes, samples)
    tree = TreeNode(attributeValue, attributeCritical)
    if attributeValue:
        attributes.remove(attributeValue)
        leftSamples, rightSamples = splitSamples(samples, attributeValue, tree.critical)
        global DEBUG
        if DEBUG:
            print len(leftSamples), len(rightSamples)
            DEBUG = False
        tree.leftChild = buildDecisionTree(leftSamples, attributes, samples)
        tree.rightChild = buildDecisionTree(rightSamples, attributes, samples)
        attributes.append(attributeValue)
        return tree
    else:
        return getPluralityClassification(samples)

def getPluralityClassification(samples):
    counter = Counter()
    for item in samples:
        counter[item.result] += 1
    return Classification(counter.most_common(1)[0][0])

def hasSameClassification(samples):
    tempSet = set()
    for item in samples:
        tempSet.add(item.result)
    return len(tempSet) == 1

def getMaxImportant(attributes, samples):
    attributeValue, attributeCritical, attributeEntropy = '', 0, 0
    for attr in attributes:
        entropy, critical = getImportantValue(samples, attr)
        if entropy > attributeEntropy:
            attributeValue = attr
            attributeEntropy = entropy
            attributeCritical = critical
    FILE.write(str(attributeEntropy) + ' ' + attributeValue + ' ' + str(attributeCritical) + ' ' + str(len(samples)) + '\n')
    return attributeValue, attributeCritical, attributeEntropy

def getImportantValue(samples, attribute):
    # sorted(samples, key=lambda x: x.values[attribute])
    criticals = list(getCriticalValues(samples, attribute))
    results = list(getRemainderValues(samples, attribute, criticals))
    return max(results, key=lambda x: x[0])

def getCriticalValues(samples, attribute):
    average = sum(map(lambda x: x.values[attribute], samples)) * 1.0 / len(samples)
    yield average
    # yield (average + samples[0].values[attribute]) / 2
    # yield (average + samples[-1].values[attribute]) / 2
    # for index in xrange(len(samples) - 1):
    #     if samples[index] != samples[index + 1]:
    #         yield (samples[index].values[attribute] +
    #             samples[index + 1].values[attribute]) / 2

def getRemainderValues(samples, attribute, criticals):
    totalLength = len(samples)
    resultCounter = Counter()
    for item in samples:
        resultCounter[item.result] += 1
    samplesEntropy = getEntropyFromCounter(resultCounter, totalLength)
    for critical in criticals:
        totalLarger, totalSmaller = 0, 0
        largerCounter, smallerCounter = Counter(), Counter()
        for item in samples:
            if item.values[attribute] >= critical:
                totalLarger += 1
                largerCounter[item.result] += 1
            else:
                totalSmaller += 1
                smallerCounter[item.result] += 1
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
    leftSamples, rightSamples = [], []
    for item in samples:
        if item.values[attribute] < critical:
            leftSamples.append(item)
        else:
            rightSamples.append(item)
    return leftSamples, rightSamples

def findPath(tree, sample):
    if isinstance(tree, Classification):
        return tree.value
    if sample.values[tree.attribute] < tree.critical:
        return findPath(tree.leftChild, sample)
    else:
        return findPath(tree.rightChild, sample)

def getRandomAttributes(attributes):
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
    for index in xrange(1):
        tempAttributes = attributes
        tree = buildDecisionTree(samples, tempAttributes, samples)
        file = open('result/result' + str(index) + '.csv', 'w')
        file.write('id,label\n')
        for i, item in enumerate(tests):
            value = findPath(tree, item)
            file.write('%d,%s\n' % (i, value))
        file.close()
        print 'Complete %d tree' % index
    end = time.time()
    print end - start
    FILE.close()