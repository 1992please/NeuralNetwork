#pragma once
#include "Matrix.h"
#include <vector>

struct FDataSet;
class TrainingData;
struct NetError
{
	double regression_error;
	double classification_error;
};
struct TrainingOutput
{
	Mat<double> Weights;
	NetError TrainingError;
	NetError ValidationError;
	NetError TestError;
};

double RandomValue(void);
NetError EvalNetworkError(FDataSet& data, Mat<double>& Weight);

struct FeedForwardOut
{
	Mat<double> net;
	Mat<double> Outputs;
};
FeedForwardOut FeedForward(Mat<double>& inputs, Mat<double>& weights, Mat<double>& bias);
Mat<double> ActivationFunc(Mat<double>& In);
Mat<double> ActivationFuncDeriv(Mat<double>& In);
double CalcClassificationError(std::vector<int>& class1, std::vector<int>& class2);
void BackPropagation(FDataSet& DataSet, Mat<double>& Weight, double eta);
TrainingOutput Train(TrainingData& Data);