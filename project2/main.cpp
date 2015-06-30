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

using namespace std;

const int LABEL_NUM = 27;

struct Sample {
    Sample(char* line, bool isTest) {
        char* tokenPtr = strtok(line, ",");
        char* tempToken = NULL;
        while (tokenPtr != NULL) {
            if (tempToken != NULL) {
                tempToken = tokenPtr;
                tokenPtr = strtok(NULL, ",");
                if (tokenPtr != NULL || isTest) {
                    this->values.push_back(atof(tempToken));
                } else {
                    this->label = atof(tempToken);
                }
            } else {
                tempToken = tokenPtr;
                tokenPtr = strtok(NULL, ",");
            }
        }
    }
    vector<double> values;
    int label;
};

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

vector<Sample> readDataFromFile(char* fileName, bool isTest) {
    FILE* file = fopen(fileName, "r");
    vector<Sample> samples;
    if (file != NULL) {
        char input[10000];
        while (!feof(file)) {
            if (fgets(input, 10000, file)) {
                if (input[0] != 'i') {
                    samples.push_back(Sample(input, isTest));
                }
            }
        }
        fclose(file);
    }
    return samples;
}

TreeNode* getPluralityClassification(const vector<Sample>& samples) {
    int labels[LABEL_NUM] = {0};
    int maxLabel = 0;
    for (int i = 0; i < samples.size(); i++) {
        labels[samples[i].label]++;
    }
    for (int i = 0; i < LABEL_NUM; i++) {
        if (labels[i] > maxLabel) {
            maxLabel = i;
        }
    }
    return new TreeNode(maxLabel, 0.0, true);
}

bool hasSameClassification(const vector<Sample>& samples) {
    set<int> tempSet;
    for (int i = 0; i < samples.size(); i++) {
        tempSet.insert(samples[i].label);
    }
    return tempSet.size() == 1;
}

vector<double> getCriticalValues(const vector<Sample>& samples, const int& attribute) {
    vector<double> criticals;
    double total = 0;
    for (int i = 0; i < samples.size(); i++) {
        total += samples[i].values[attribute];
    }
    criticals.push_back(total * 1.0 / samples.size());
    return criticals;
}

double getEntropy(const int counter[], const int& length) {
    double entropy = 0;
    for (int i = 1; i < LABEL_NUM; i++) {
        if (counter[i] != 0) {
            double temp = counter[i] * 1.0 / length;
            entropy -= temp * log(temp) / log(2);
        }
    }
    return entropy;
}

void getImportantValue(double& critical, double& entropy, vector<Sample> samples, int attribute) {
    vector<double> criticals = getCriticalValues(samples, attribute);
    int totalLength = samples.size();
    int labelCounter[LABEL_NUM] = {0};
    for (int i = 0; i < samples.size(); i++) {
        labelCounter[samples[i].label]++;
    }
    double totalEntropy = getEntropy(labelCounter, totalLength);
    for (int i = 0; i < criticals.size(); i++) {
        int totalLarger = 0, totalSmaller = 0;
        int largerCounter[LABEL_NUM] = {0}, smallerCounter[LABEL_NUM] = {0};
        for (int j = 0; j < samples.size(); j++) {
            if (samples[j].values[attribute] >= criticals[i]) {
                largerCounter[samples[j].label]++;
                totalLarger++;
            } else {
                smallerCounter[samples[j].label]++;
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

void getMaxImportant(int& attribute, double& critical, const vector<Sample>& samples, const set<int>& attributes) {
    attribute = 0, critical = 0.0;
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

void splitSamples(vector<Sample>& leftSamples, vector<Sample>& rightSamples,
    const vector<Sample>& samples, const int& attribute, const int& critical) {
    for (int i = 0; i < samples.size(); i++) {
        if (samples[i].values[attribute] < critical) {
            leftSamples.push_back(samples[i]);
        } else {
            rightSamples.push_back(samples[i]);
        }
    }
}

TreeNode* buildDecisionTree(const vector<Sample>& samples,
    set<int> attributes, const vector<Sample>& parentSamples) {
    if (samples.size() < 5) {
        return getPluralityClassification(parentSamples);
    }
    if (hasSameClassification(samples)) {
        return new TreeNode(samples[0].label, 0.0, true);
    }
    if (attributes.size() == 0) {
        return getPluralityClassification(parentSamples);
    }
    int attribute;
    double critical;
    getMaxImportant(attribute, critical, samples, attributes);
    TreeNode* tree = new TreeNode(attribute, critical, false);
    attributes.erase(attribute);
    vector<Sample> leftSamples, rightSamples;
    splitSamples(leftSamples, rightSamples, samples, attribute, critical);
    tree->leftChild = buildDecisionTree(leftSamples, attributes, samples);
    tree->rightChild = buildDecisionTree(rightSamples, attributes, samples);
    attributes.insert(attribute);
    return tree;
}

int findPath(TreeNode* tree, Sample sample) {
    if (tree->isLeaf) {
        return tree->attribute;
    }
    if (sample.values[tree->attribute] < tree->critical) {
        return findPath(tree->leftChild, sample);
    }
    return findPath(tree->rightChild, sample);
}

void writeToFile(vector<int> results) {
    ofstream fout("result.csv");
    fout << "id,label" << endl;
    for (int i = 0; i < results.size(); i++) {
        fout << i << "," << results[i] << endl;
    }
    fout.close();
}

int main() {
    vector<Sample> samples = readDataFromFile("train.csv", false);
    vector<Sample> tests = readDataFromFile("test.csv", false);
    set<int> attributes;
    for (int i = 0; i < samples[0].values.size(); i++) {
        attributes.insert(i);
    }
    TreeNode* tree = buildDecisionTree(samples, attributes, samples);
    vector<int> results;
    for (int i = 0; i < tests.size(); i++) {
        results.push_back(findPath(tree, tests[i]));
    }
    writeToFile(results);
    tree->clear();
    delete tree;
    return 0;
}
