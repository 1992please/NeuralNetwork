#pragma once

#include "Matrix.h"
#include <vector>

struct FCostFunctionOut
{
	double J;
	Mat<double> Theta1_Grad;
	Mat<double> Theta2_Grad;
};

class TrainingData;

class NNeuralNet
{
private:
	Mat<double> X;
	Mat<double> Y;
	Mat<double> Theta1;
	Mat<double> Theta2;
	uint16_t input_layer_size;
	uint16_t hidden_layer_size;
	uint16_t num_labels ;



	static Mat<double> ConvertClassToOutput(Mat<double>& In, uint16_t NoOfLabels);
	static Mat<double> RandInitializeWeights(uint16_t L_in, size_t L_out);
	FCostFunctionOut nnCostFunction(double lambda);
	double CalcTrainingClassificationError();
public:
	NNeuralNet(TrainingData* Data);
	Mat<double> Predict(Mat<double>& X);
	static double CalcClassificationError(std::vector<int>& class1, std::vector<int>& class2);
	static double CalcClassificationError(Mat<double>& Output1, Mat<double>& Output2);

	double Train(const uint32_t epoch_max = 100,const double Learning_Rate = .1, const double lambda = 0);
};

