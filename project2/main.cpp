#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <set>
#include <cmath>
#include <ctime>
#include <thread>
#include <algorithm>

using namespace std;

char TRAIN_FILE[] = "train.csv";
char TEST_FILE[] = "test.csv";

const int LABEL_NUM = 27;
const int ATTRIBUTE_NUM = 617;
double ATTRIBUTE_RATE = 0.12;
double SAMPLE_RATE = 0.6;
int THREAD_NUM = 4;
int TREE_NUM = 25;
bool isMultiThread = true;

struct Sample {
    Sample(char* line, bool isTest) {
        char* tokenPtr = strtok(line, ",");
        char* tempToken = NULL;
        int count = 0;
        while (tokenPtr != NULL) {
            if (tempToken != NULL) {
                tempToken = tokenPtr;
                tokenPtr = strtok(NULL, ",");
                if (tokenPtr != NULL || isTest) {
                    this->values[count++] = atof(tempToken);
                } else {
                    this->label = atof(tempToken);
                }
            } else {
                tempToken = tokenPtr;
                tokenPtr = strtok(NULL, ",");
            }
        }
    }
    double values[ATTRIBUTE_NUM];
    int label;
};

vector<Sample> allSamples;
vector<Sample> allTests;
vector<int>* results;

class TreeNode {
public:
    TreeNode(int attribute, double critical, bool isLeaf) {
        this->attribute = attribute;
        this->critical = critical;
        this->isLeaf = isLeaf;
        this->leftChild = NULL;
        this->rightChild = NULL;
    }
    void clear() {
        if (this->leftChild) {
            this->leftChild->clear();
        }
        if (this->rightChild) {
            this->rightChild->clear();
        }
        delete this->rightChild;
        delete this->leftChild;
    }
    int attribute;
    double critical;
    bool isLeaf;
    TreeNode* leftChild;
    TreeNode* rightChild;
};

TreeNode* getPluralityClassification(const vector<int>& samples) {
    int labels[LABEL_NUM] = {0};
    int maxLabel = 0;
    for (int i = 0; i < samples.size(); i++) {
        labels[allSamples[samples[i]].label]++;
    }
    for (int i = 0; i < LABEL_NUM; i++) {
        if (labels[i] > maxLabel) {
            maxLabel = i;
        }
    }
    return new TreeNode(maxLabel, 0.0, true);
}

bool hasSameClassification(const vector<int>& samples) {
    int lastLabel = allSamples[samples[0]].label;
    for (int i = 0; i < samples.size(); i++) {
        int tempLabel = allSamples[samples[i]].label;
        if (lastLabel != tempLabel) {
            return false;
        }
        lastLabel = tempLabel;
    }
    return true;
}

vector<double> getCriticalValues(const vector<int>& samples, const int& attribute) {
    vector<double> criticals;
    double total = 0.0;
    for (int i = 0; i < samples.size(); i++) {
        total += allSamples[samples[i]].values[attribute];
    }
    // criticals.push_back(allSamples[samples[(samples.size() - 1) / 8]].values[attribute]);
    // criticals.push_back(allSamples[samples[(samples.size() - 1) * 2 / 8]].values[attribute]);
    // criticals.push_back(allSamples[samples[(samples.size() - 1) * 3 / 8]].values[attribute]);
    // criticals.push_back(allSamples[samples[(samples.size() - 1) * 4 / 8]].values[attribute]);
    // criticals.push_back(allSamples[samples[(samples.size() - 1) * 5 / 8]].values[attribute]);
    // criticals.push_back(allSamples[samples[(samples.size() - 1) * 6 / 8]].values[attribute]);
    // criticals.push_back(allSamples[samples[(samples.size() - 1) * 7 / 8]].values[attribute]);
    // criticals.push_back(allSamples[samples[(samples.size() - 1) * 8 / 8]].values[attribute]);
    criticals.push_back(total / samples.size());
    return criticals;
}

double getEntropy(const int counter[], const int& length) {
    double entropy = 0;
    for (int i = 0; i < LABEL_NUM; i++) {
        if (counter[i] != 0) {
            double temp = counter[i] * 1.0 / length;
            entropy -= temp * log(temp) / log(2);
        }
    }
    return entropy;
}

void getImportantValue(double& critical, double& entropy, vector<int> samples, int attribute) {
    vector<double> criticals = getCriticalValues(samples, attribute);
    int totalLength = samples.size();
    int labelCounter[LABEL_NUM] = {0};
    for (int i = 0; i < samples.size(); i++) {
        labelCounter[allSamples[samples[i]].label]++;
    }
    double totalEntropy = getEntropy(labelCounter, totalLength);
    for (int i = 0; i < criticals.size(); i++) {
        int totalLarger = 0, totalSmaller = 0;
        int largerCounter[LABEL_NUM] = {0}, smallerCounter[LABEL_NUM] = {0};
        for (int j = 0; j < samples.size(); j++) {
            if (allSamples[samples[j]].values[attribute] >= criticals[i]) {
                largerCounter[allSamples[samples[j]].label]++;
                totalLarger++;
            } else {
                smallerCounter[allSamples[samples[j]].label]++;
                totalSmaller++;
            }
        }
        double remainder = 0;
        if (totalSmaller != 0) {
            remainder += totalSmaller * 1.0 / totalLength * getEntropy(smallerCounter, totalSmaller);
        }
        if (totalLarger != 0) {
            remainder += totalLarger * 1.0 / totalLength * getEntropy(largerCounter, totalLarger);
        }
        if (totalEntropy - remainder >= entropy) {
            entropy = totalEntropy - remainder;
            critical = criticals[i];
        }
    }
}

void getMaxImportant(int& attribute, double& critical, const vector<int>& samples, const set<int>& attributes) {
    double entropy = -1;
    for (set<int>::iterator it = attributes.begin(); it != attributes.end(); it++) {
        double tempCritical = 0.0, tempEntropy = 0.0;
        getImportantValue(tempCritical, tempEntropy, samples, *it);
        if (tempEntropy > entropy) {
            entropy = tempEntropy;
            attribute = *it;
            critical = tempCritical;
        }
    }
}

void splitSamples(vector<int>& leftSamples, vector<int>& rightSamples,
    const vector<int>& samples, const int& attribute, const double& critical) {
    for (int i = 0; i < samples.size(); i++) {
        if (allSamples[samples[i]].values[attribute] < critical) {
            leftSamples.push_back(samples[i]);
        } else {
            rightSamples.push_back(samples[i]);
        }
    }
}

TreeNode* buildDecisionTree(const vector<int>& sampleIndexs,
    set<int> attributes, const vector<int>& parentSampleIndexs) {
    if (sampleIndexs.size() == 0) {
        return getPluralityClassification(parentSampleIndexs);
    }
    if (hasSameClassification(sampleIndexs)) {
        return new TreeNode(allSamples[sampleIndexs[0]].label, 0.0, true);
    }
    if (attributes.size() == 0) {
        return getPluralityClassification(parentSampleIndexs);
    }
    int attribute = 0;
    double critical = 0.0;
    getMaxImportant(attribute, critical, sampleIndexs, attributes);
    TreeNode* tree = new TreeNode(attribute, critical, false);
    attributes.erase(attribute);
    vector<int> leftSamples, rightSamples;
    splitSamples(leftSamples, rightSamples, sampleIndexs, attribute, critical);
    tree->leftChild = buildDecisionTree(leftSamples, attributes, sampleIndexs);
    tree->rightChild = buildDecisionTree(rightSamples, attributes, sampleIndexs);
    attributes.insert(attribute);
    return tree;
}


vector<Sample> readDataFromFile(char* fileName, bool isTest) {
    FILE* file = fopen(fileName, "r");
    vector<Sample> samples;
    if (file != NULL) {
        char input[100000];
        while (!feof(file)) {
            if (fgets(input, 100000, file)) {
                if (input[0] != 'i') {
                    samples.push_back(Sample(input, isTest));
                }
            }
        }
        fclose(file);
    }
    return samples;
}

void writeToFile(const vector<int> results[]) {
    ofstream fout("result.csv");
    const int ids = allTests.size();
    int counter[ids][LABEL_NUM], totalTree = TREE_NUM * THREAD_NUM;
    memset(counter, 0, sizeof(counter));
    for (int i = 0; i < totalTree; i++) {
        for (int j = 0; j < results[i].size(); j++) {
            counter[j][results[i][j]]++;
        }
    }
    fout << "id,label" << endl;
    for (int i = 0; i < ids; i++) {
        int maxLabel = 0, maxCount = 0;
        for (int j = 0; j < LABEL_NUM; j++) {
            if (counter[i][j] > maxCount) {
                maxLabel = j;
                maxCount = counter[i][j];
            }
        }
        fout << i << "," << maxLabel << endl;
    }
    fout.close();
}

int findPath(TreeNode* tree, Sample& sample) {
    if (tree->isLeaf) {
        return tree->attribute;
    }
    if (sample.values[tree->attribute] < tree->critical) {
        return findPath(tree->leftChild, sample);
    }
    return findPath(tree->rightChild, sample);
}

set<int> getRandomAttributes() {
    set<int> attributes;
    int targetSize = ATTRIBUTE_NUM * ATTRIBUTE_RATE;
    while (attributes.size() != targetSize) {
        attributes.insert(rand() % ATTRIBUTE_NUM);
    }
    return attributes;
}

vector<int> getRandomSampleIndexs() {
    vector<int> samples;
    int totalSampleSize = allSamples.size();
    int targetSize = SAMPLE_RATE * totalSampleSize;
    for (int i = 0; i < targetSize; i++) {
        samples.push_back(rand() % totalSampleSize);
    }
    return samples;
}

void randomForestClassify() {
    for (int i = 0; i < TREE_NUM; i++) {
        vector<int> result;
        set<int> attributes = getRandomAttributes();
        vector<int> samples = getRandomSampleIndexs();
        TreeNode* tree = buildDecisionTree(samples, attributes, samples);
        for (int j = 0; j < allTests.size(); j++) {
            result.push_back(findPath(tree, allTests[j]));
        }
        results[i] = result;
        tree->clear();
        delete tree;
    }
}

void randomForestClassifyThread(int index) {
    int threadId = index;
    for (int i = 0; i < TREE_NUM; i++) {
        vector<int> result;
        set<int> attributes = getRandomAttributes();
        vector<int> samples = getRandomSampleIndexs();
        TreeNode* tree = buildDecisionTree(samples, attributes, samples);
        for (int j = 0; j < allTests.size(); j++) {
            result.push_back(findPath(tree, allTests[j]));
        }
        results[threadId * TREE_NUM + i] = result;
        tree->clear();
        delete tree;
    }
}

void multiRandomForestClassify() {
    thread threads[TREE_NUM];
    for (int i = 0; i < THREAD_NUM; i++) {
        threads[i] = thread(randomForestClassifyThread, i);
    }
    for (int i = 0; i < THREAD_NUM; i++) {
        threads[i].join();
    }
}

void initializeData(int argc, char* argv[]) {
    allSamples = readDataFromFile(TRAIN_FILE, false);
    allTests = readDataFromFile(TEST_FILE, true);
    srand((unsigned)time(NULL));
    if (argc > 0) {
        THREAD_NUM = (int)atof(argv[1]);
    }
    if (argc > 1) {
        TREE_NUM = (int)atof(argv[2]);
    }
    if (argc > 2) {
        ATTRIBUTE_RATE = atof(argv[3]);
    }
    if (argc > 3) {
        SAMPLE_RATE = atof(argv[4]);
    }
    results = new vector<int>[TREE_NUM * THREAD_NUM];
}

int main(int argc, char* argv[]) {
    initializeData(argc, argv);
    multiRandomForestClassify();
    // randomForestClassify();
    writeToFile(results);
    delete []results;
    return 0;
}
