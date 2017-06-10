#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include "Matrix.h"

struct FDataSet
{
	Mat<double> inputs;
	Mat<double> outputs;
	Mat<double> bias;
	std::vector<int> classes;
	size_t count;

	void Show()
	{
		inputs.Show("Inputs");
		outputs.Show("Outputs");
		bias.Show("bias");
	}
	static std::vector<int> ConvertOuputToClass(Mat<double>& In);
	static Mat<double> ConvertClassToOutput(std::vector<int>& In);
};

class TrainingData
{
public:
	TrainingData();
	FDataSet TrainingSet;
	FDataSet ValidationSet;
	FDataSet TestSet;
	// Returns the number of input values read from the file:
private:
	void UpdateDataSet(FDataSet &InDataSet, char*const filename);

};

