#pragma once
#include "Eigen\Dense"

struct FCostFunctionOut
{
	double J;
	Eigen::MatrixXd Theta1_Grad;
	Eigen::MatrixXd Theta2_Grad;
};

class NNetwork
{
private:
	Eigen::MatrixXd X;
	Eigen::MatrixXd Y;


	Eigen::MatrixXd Theta1;
	Eigen::MatrixXd Theta2;
	uint64_t input_layer_size;
	uint64_t hidden_layer_size;
	uint64_t num_labels;

	FCostFunctionOut nnCostFunction(Eigen::MatrixXd& _Theta1, Eigen::MatrixXd& _Theta2, double lambda);
	double CalcTrainingError();
public:
	Eigen::MatrixXd Xtest;
	Eigen::MatrixXd Ytest;
	NNetwork(Eigen::MatrixXd& _X, Eigen::MatrixXd&  _Y, double TestRatio = 0.3);

	~NNetwork();
	double Train(const uint64_t epoch_max, const double Learning_Rate, const double lambda);
	Eigen::MatrixXd Predict(Eigen::MatrixXd& _X);

	static Eigen::MatrixXd RandInitializeWeights(uint64_t L_in, size_t L_out);
	static Eigen::MatrixXd ConvertClassToOutput(Eigen::MatrixXd& In, uint16_t NoOfLabels);
	static Eigen::MatrixXd GetMatrix(char*const filename);

	void Save(char* filePath);
};

class FFNetwork
{
private:
	Eigen::MatrixXd Theta1;
	Eigen::MatrixXd Theta2;
public:
	FFNetwork(char* const filename);
	Eigen::RowVectorXd Predict(Eigen::RowVectorXd& _X);
	Eigen::MatrixXd Predict(Eigen::MatrixXd& _X);
};
