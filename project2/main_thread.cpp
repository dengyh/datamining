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

using namespace std;

char TRAIN_FILE[] = "train.csv";
char TEST_FILE[] = "test.csv";

const int LABEL_NUM = 27;
const int ATTRIBUTE_NUM = 617;
const int THREAD_NUM = 4;
const int EVERY_TREE_ATTRIBUTE_NUM = 24;
const int EVERY_TREE_SAMPLE_NUM = 6283 * 0.6; // MAX 6238
const int TREE_NUM = 100; // Every Thread

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
vector<int> results[TREE_NUM];

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
    // criticals.push_back(allSamples[samples[samples.size() / 2]].values[attribute]);
    // criticals.push_back(allSamples[samples[samples.size() / 4]].values[attribute]);
    // criticals.push_back(allSamples[samples[samples.size() * 3 / 4]].values[attribute]);
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
    double entropy = 0.0;
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
    int counter[ids][LABEL_NUM];
    memset(counter, 0, sizeof(counter));
    for (int i = 0; i < TREE_NUM; i++) {
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
    while (attributes.size() != EVERY_TREE_ATTRIBUTE_NUM) {
        attributes.insert(rand() % ATTRIBUTE_NUM);
    }
    return attributes;
}

vector<int> getRandomSampleIndexs() {
    vector<int> samples;
    int totalSampleSize = allSamples.size();
    for (int i = 0; i < EVERY_TREE_SAMPLE_NUM; i++) {
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
        cout << "Run tree " << i << " finish" << endl;
        results[i] = result;
        tree->clear();
        delete tree;
    }
}

void* randomForestClassifyThread(void* index) {
    int number = *(int*)index;
    vector<int> result;
    set<int> attributes = getRandomAttributes();
    vector<int> samples = getRandomSampleIndexs();
    TreeNode* tree = buildDecisionTree(samples, attributes, samples);
    for (int j = 0; j < allTests.size(); j++) {
        result.push_back(findPath(tree, allTests[j]));
    }
    cout << number << endl;
    results[number] = result;
    tree->clear();
    delete tree;
}

void multiRandomForestClassify() {
    pthread_t thread[TREE_NUM];
    for (int i = 0; i < TREE_NUM; i++) {
        int ret = pthread_create(&thread[i], NULL, randomForestClassifyThread, &i);
    }
    for (int i = 0; i < TREE_NUM; i++) {
        pthread_join(thread[i], NULL);
    }
}

void initializeData() {
    allSamples = readDataFromFile(TRAIN_FILE, false);
    allTests = readDataFromFile(TEST_FILE, true);
}

int main() {
    clock_t start, finish;
    start = clock();
    initializeData();
    // multiRandomForestClassify();
    randomForestClassify();
    writeToFile(results);
    finish = clock();
    cout << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}
