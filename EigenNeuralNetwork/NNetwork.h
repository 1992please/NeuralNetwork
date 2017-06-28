#pragma once
#include "Eigen\Dense"

struct FCostFunctionOut
{
	double J;
	Eigen::MatrixXd Theta1_Grad;
	Eigen::MatrixXd Theta2_Grad;
};

struct FFeatureNormalizeOut
{
	Eigen::MatrixXd X_norm;
	// the mean
	Eigen::RowVectorXd mu;
	// standard deviation
	Eigen::RowVectorXd sigma;
};

struct FPCAOut
{
	Eigen::MatrixXd U;
	Eigen::MatrixXd S;
};


class NNetwork
{
private:
	Eigen::RowVectorXd Mu;
	Eigen::RowVectorXd Sigma;
	Eigen::MatrixXd U;
	Eigen::MatrixXd X_real;
	Eigen::MatrixXd X;
	Eigen::MatrixXd Y;

	Eigen::MatrixXd Theta1;
	Eigen::MatrixXd Theta2;
	uint64_t input_layer_size;
	uint64_t hidden_layer_size;
	uint64_t num_labels;
	bool bUsingPCA;


	static FFeatureNormalizeOut FeatureNormalize(const Eigen::MatrixXd& _X);
	static FPCAOut PCA(const Eigen::MatrixXd& _X);
	static Eigen::MatrixXd RandInitializeWeights(uint64_t L_in, size_t L_out);

	Eigen::MatrixXd ApplyPCA(const Eigen::MatrixXd _X, const double VarianceRetained);
	double CalcTrainingError();
public:
	Eigen::MatrixXd Xtest;
	Eigen::MatrixXd Ytest;


	NNetwork(Eigen::MatrixXd& _X, Eigen::MatrixXd&  _Y, double PCA_Retained_Variance = 1, double TestRatio = 0.3);

	~NNetwork();
	double Train(const uint64_t epoch_max, const double Learning_Rate, const double lambda);
	Eigen::MatrixXd Predict(Eigen::MatrixXd& _XIn);
	void Save(char* filePath);

	static FCostFunctionOut nnCostFunction(const Eigen::MatrixXd& _X, const Eigen::MatrixXd& _Y, const Eigen::MatrixXd& _Theta1, const  Eigen::MatrixXd& _Theta2, const double lambda);
	static Eigen::MatrixXd ConvertClassToOutput(Eigen::MatrixXd& In, uint16_t NoOfLabels);
	static Eigen::MatrixXd GetMatrix(char*const filename);

	// Get Matrix from Opened file
	static void SavetMatrix(const Eigen::MatrixXd& Mat, std::ofstream&  file);
};

class FFNetwork
{
private:
	Eigen::RowVectorXd Mu;
	Eigen::RowVectorXd Sigma;
	Eigen::MatrixXd U;
	Eigen::MatrixXd Theta1;
	Eigen::MatrixXd Theta2;
public:
	FFNetwork(char* const filename);
	Eigen::RowVectorXd Predict(Eigen::RowVectorXd& _X);
	Eigen::MatrixXd Predict(Eigen::MatrixXd& _X);

	static Eigen::MatrixXd GetMatrix(std::ifstream& file);
};