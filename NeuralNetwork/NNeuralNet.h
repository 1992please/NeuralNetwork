#pragma once
#include "Matrix.h"
#include <vector>


struct FDataSet;
class TrainingData;
static Mat<double> ConvertClassToOutput(Mat<uint16_t>& In);
static Mat<double> RandInitializeWeights(uint16_t L_in, size_t L_out);
static std::vector<int> predict(Mat<double> Theta1, Mat<double> Theta2, Mat<double> X);
struct FCostFunctionOut
{
	double J;
	Mat<double> grad;
};
FCostFunctionOut nnCostFunction(Mat<double> Theta1, Mat<double> Theta2, uint16_t num_labels, Mat<double> X, Mat<double> Y, double lambda);

void Train(TrainingData& Data);