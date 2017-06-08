#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include "Matrix.h"

struct FSet
{
	Mat<double> inputs;
	Mat<double> outputs;
	Mat<int> classes;
	unsigned count;
	std::vector<unsigned> bias;
};

class TrainingData
{
public:
	TrainingData(char*const filename);

	// Returns the number of input values read from the file:
	void UpdateDataSet();

private:
	void FillLines();
	void FillSets();
	FSet DataSet;
	std::vector<std::string> lines;
	char* filename;
};

