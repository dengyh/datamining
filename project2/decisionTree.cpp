#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <pthread.h>

using namespace std;

string TRAIN = "train.csv";
string TEST = "test.csv";
const int attributesNum = 617;
const int treesNum = 50;
const int examplesNum = 6238;

struct Example {
	double attrValues[attributesNum];
	int label;
};

struct AttrBound {
	int importantAttr;
	double importantBound;
};

struct Argument {
	double sampleRate;
	double attributeRate;
	int index;
};

vector<Example> Examples;
vector<Example> TestExamples;

struct TreeNode {
	TreeNode* leftChild;
	TreeNode* rightChild;
	bool isLeaf;
	int attribute;
	double bound;
	int value;
	TreeNode() {
		value = -1;
		isLeaf = false;
	}
	~TreeNode() {
		if(leftChild != NULL) {
			delete this->leftChild;
		}
		if(rightChild != NULL) {
			delete this->rightChild;
		}
		delete this;
	}
};

TreeNode* randomForest[treesNum];

vector<Example> getData(string filename) {
	FILE* trainFile = fopen(filename.c_str(), "r");
	string line;
	int exampleNum = 0;
	vector<Example> examples;
	char buffer[10000];
	bool isFirst = true;
	while (fgets(buffer, 10000, trainFile)) {
		if (isFirst) {
			isFirst = false;
			continue;
		}
		Example example;
		int Label;
		char* buf;
		int attr = 0;
		buf = strtok(buffer, ",");
		buf = strtok(NULL, ",");
		while (attr < attributesNum) {
			example.attrValues[attr++] = atof(buf);
			buf = strtok(NULL, ",");
		}
		if (buf != NULL)
			example.label = atoi(buf);
		examples.push_back(example);
	}
	return examples;
}

int plurality_value(vector<int> exampleIndexs) {
	int count[27];
	memset(count, 0, sizeof(count));
	for (int i = 0; i < exampleIndexs.size(); i++) {
		count[Examples[exampleIndexs[i]].label] ++;
	}
	int max = 0;
	int maxLabel = -1;
	for (int i = 0; i < 27; i++) {
		if (max < count[i]) {
			max = count[i];
			maxLabel = i;
		}
	}
	return maxLabel;
}

int hasSameClassification(vector<int> exampleIndexs) {
	int count[27];
	memset(count, 0, sizeof(count));
	for (int i = 0; i < exampleIndexs.size(); i++) {
		count[Examples[exampleIndexs[i]].label] ++;
	}
	for (int i = 0; i < 27; i++) {
		if (count[i] > double(exampleIndexs.size()) * 0.95)
			return i;
	}
	return -1;
}

double Entropy(vector<int> exampleIndexs) {
	int classCount[27];
	int total = exampleIndexs.size();
	for (int i = 0; i < 27; i++)
		classCount[i] = 0;
	double sum = 0.0;
	for (int i = 0; i < total; i++) {
		classCount[Examples[exampleIndexs[i]].label] ++;
	}
	for (int i = 0; i < 27; i++) {
		if (classCount[i] == 0)
			continue;
		sum -= double(classCount[i]) / total * (log(double(classCount[i]) / total) / log(2));
	}
	return sum;
}

double Gain(double bound, int attr, vector<int> exampleIndexs) {
	double Es = Entropy(exampleIndexs);
	int exLength = exampleIndexs.size();
	vector<int> exs1, exs2;
	for (int i = 0; i < exLength; i++) {
		if (Examples[exampleIndexs[i]].attrValues[attr] <= bound)
			exs1.push_back(exampleIndexs[i]);
		else
			exs2.push_back(exampleIndexs[i]);
	}
	double gain = Es - double(exs1.size()) / double(exLength) * Entropy(exs1) - double(exs2.size()) / double(exLength) * Entropy(exs2);
	return gain;
}

double getBound(vector<int> exampleIndexs, int attr) {
	double bound = 0.0;
	for (int i = 0; i < exampleIndexs.size(); i++)
		bound += Examples[exampleIndexs[i]].attrValues[attr];
	bound /= exampleIndexs.size();
	return bound;
}

AttrBound getImportantAttribute(vector<int> exampleIndexs, vector<int> attributes) {
	double maxGain = -9999;
	double importBound = -9999;
	int importAttr = -1;
	for (int i = 0; i < attributes.size(); i++) {
		double tmpBound = getBound(exampleIndexs, attributes[i]);
		double tmpGain = Gain(tmpBound, attributes[i], exampleIndexs);
		if (maxGain < tmpGain) {
			maxGain = tmpGain;
			importBound = tmpBound;
			importAttr = attributes[i];
		}
	}
	AttrBound attrBound;
	attrBound.importantAttr = importAttr;
	attrBound.importantBound = importBound;
	return attrBound;
}

TreeNode* decision_tree_learning(vector<int> exampleIndexs, vector<int> attributes) {
	TreeNode* decisionTree = new TreeNode();

	int isHSC = hasSameClassification(exampleIndexs);
	if (isHSC != -1) {
		decisionTree->isLeaf = true;
		decisionTree->value = isHSC;
		return decisionTree;
	}
	if (exampleIndexs.size() < 5 || attributes.size() == 0) {
		decisionTree->isLeaf = true;
		decisionTree->value = plurality_value(exampleIndexs);
		return decisionTree;
	}
	AttrBound attrBound = getImportantAttribute(exampleIndexs, attributes);
	
	decisionTree->attribute = attrBound.importantAttr;
	decisionTree->bound = attrBound.importantBound;
	
	vector<int> attributes_new;
	for (int i = 0; i < attributes.size(); i++) {
		if (attributes[i] != attrBound.importantAttr)
			attributes_new.push_back(attributes[i]);
	}

	vector<int> exsLeft, exsRight;
	for (int i = 0; i < exampleIndexs.size(); i++) {
		if (Examples[exampleIndexs[i]].attrValues[attrBound.importantAttr] <= attrBound.importantBound)
			exsLeft.push_back(exampleIndexs[i]);
		else
			exsRight.push_back(exampleIndexs[i]);
	}
	//Left subtree
    	decisionTree->leftChild = decision_tree_learning(exsLeft, attributes_new);
	//Right subtree
	decisionTree->rightChild = decision_tree_learning(exsRight, attributes_new);
	return decisionTree;
}

int getTreeResult(Example example, TreeNode* decisionTree) {
	if (decisionTree->isLeaf)
		return decisionTree->value;
	else if (example.attrValues[decisionTree->attribute] <= decisionTree->bound)
		return getTreeResult(example, decisionTree->leftChild);
	else
		return getTreeResult(example, decisionTree->rightChild);

}

vector<int> selectSample(int ExampleLength, int SampleLength) {
	vector<int> sample;
	for (int i = 0; i < SampleLength; i++)
		sample.push_back(rand() % ExampleLength);
	return sample;
}

void generateRandomForest(double SampleRate, double AttributeRate) {
	for (int i = 0; i < treesNum; i++) {
		vector<int> sample = selectSample(examplesNum, SampleRate * examplesNum);
		vector<int> attributes = selectSample(attributesNum, AttributeRate * attributesNum);
		randomForest[i] = decision_tree_learning(sample, attributes);
		cout << "Tree " << i + 1 << " is Completed." << endl;
	}
}

void* decision_tree_learning_multi(void* arg) {
	Argument* sampleAndAttrs = (Argument*)arg;
	double SampleRate = sampleAndAttrs->sampleRate;
	double AttributeRate = sampleAndAttrs->attributeRate;
	int Index = sampleAndAttrs->index;
	vector<int> sample = selectSample(examplesNum, SampleRate * examplesNum);
	vector<int> attributes = selectSample(attributesNum, AttributeRate * attributesNum);
	randomForest[Index] = decision_tree_learning(sample, attributes);
	cout << "Tree " << Index << " is Completed." << endl;
}

void generateRandomForestMultiThread(double SampleRate, double AttributeRate) {
	pthread_t thread[treesNum];
	Argument arg[treesNum];
	for (int i = 0; i < treesNum; i++) {
		arg[i].sampleRate = SampleRate;
		arg[i].attributeRate = AttributeRate;
		arg[i].index = i;
		int ret = pthread_create(&thread[i], NULL, decision_tree_learning_multi, &arg[i]);
	}
	for (int i = 0; i < treesNum; i ++) {
		pthread_join(thread[i], NULL);
	}
}

int getForestResult(Example example) {
	int count[27];
	memset(count, 0, sizeof(count));
	for (int i = 0; i < treesNum; i++) {
		count[getTreeResult(example, randomForest[i])] ++;
	}
	int max = -1;
	int label = -1;
	for (int i = 0; i < 27; i++) {
		if (max < count[i]) {
			max = count[i];
			label = i;
		}
	}
	return label;
}

void generateResult() {
	ofstream out("result.csv");
	out << "id,label" << endl;
	for (int i = 0; i < TestExamples.size(); i++) {
		out << i << "," << getForestResult(TestExamples[i]) << endl;
	}
}

int main() {
	clock_t start, finish;
    start = clock();
	Examples = getData(TRAIN);
	TestExamples = getData(TEST);
	generateRandomForest(0.8, 0.5);
	cout << "Forest is Completed." << endl;
	generateResult();
	finish = clock();
    cout << (double)(finish - start) / CLOCKS_PER_SEC << "s" << endl;
}