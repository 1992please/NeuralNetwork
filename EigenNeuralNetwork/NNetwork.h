#pragma once
#include "Eigen\Dense"

using namespace  Eigen;

struct FCostFunctionOut
{
	double J;
	MatrixXd Theta1_Grad;
	MatrixXd Theta2_Grad;
};

class NNetwork
{
private:
	MatrixXd X;
	MatrixXd Y;
	MatrixXd Xtest;
	MatrixXd Ytest;

	MatrixXd Theta1;
	MatrixXd Theta2;
	uint64_t input_layer_size;
	uint64_t hidden_layer_size;
	uint64_t num_labels;

	FCostFunctionOut nnCostFunction(MatrixXd& _Theta1, MatrixXd& _Theta2, double lambda);
	double CalcTrainingError();
public:
	NNetwork(MatrixXd& _X, MatrixXd&  _Y, double TestRatio = 0.3);
	~NNetwork();
	double Train(const uint64_t epoch_max, const double Learning_Rate, const double lambda);
	MatrixXd Predict(MatrixXd& _X);

	static MatrixXd RandInitializeWeights(uint64_t L_in, size_t L_out);
	static MatrixXd ConvertClassToOutput(MatrixXd& In, uint16_t NoOfLabels);

	static Eigen::MatrixXd GetMatrix(char*const filename);
};

