#pragma once
#include "Matrix.h"
#include <vector>


struct FDataSet;
class TrainingData;
static double CalcClassificationError(std::vector<int>& class1, std::vector<int>& class2);
static Mat<double> ConvertClassToOutput(Mat<double>& In, uint16_t NoOfLabels);
static Mat<double> RandInitializeWeights(uint16_t L_in, size_t L_out);
std::vector<int> predict(Mat<double>& Theta1, Mat<double>& Theta2, Mat<double>& X);
struct FCostFunctionOut
{
	double J;
	Mat<double> Theta1_Grad;
	Mat<double> Theta2_Grad;
};
FCostFunctionOut nnCostFunction(Mat<double>& Theta1, Mat<double>& Theta2, uint16_t num_labels, Mat<double>& X, Mat<double>& Y, double lambda);


void Train(TrainingData& Data);