#include "NNetwork.h"
#include "Assert.h"
#include <fstream>
#include <sstream>
#include <iostream>

static inline double RandomValue(void) { return rand() / double(RAND_MAX); }
static inline double sigmoid(const double In) { return 1.0 / (1.0 + exp(-In)); }
static inline double sigmoidGradient(const double In) { const double sig = sigmoid(In); return sig *(1 - sig); }
static inline double square(double In) { return In * In; }

using namespace Eigen;

NNetwork::NNetwork(MatrixXd& _X, MatrixXd&  _Y, double PCA_Retained_Variance, double TestRatio)
{
	ASSERT(_X.rows() == _Y.rows());
	// shuffle the x/y matrixes
	const Eigen::Index m = _X.rows();

	MatrixXd XPCA = ApplyPCA(_X, PCA_Retained_Variance);

	PermutationMatrix<Dynamic, Dynamic> perm(m);
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
	const MatrixXd XFull = perm * XPCA;
	const MatrixXd YFull = perm * _Y;
	const int testSize = (int)(TestRatio * m);
	const int trainSize = (int)(m - testSize);


	X = XFull.topRows(trainSize);
	Y = YFull.topRows(trainSize);

	Xtest = XFull.bottomRows(testSize);
	Ytest = YFull.bottomRows(testSize);

	input_layer_size = X.cols();
	num_labels = Y.cols();
	hidden_layer_size = num_labels;
}

NNetwork::~NNetwork()
{
}

double NNetwork::CalcTrainingError()
{
	if (Xtest.rows() > 0)
	{
		MatrixXd YPred = Predict(Xtest);

		return (YPred - Ytest).squaredNorm();
	}
	else
	{
		MatrixXd YPred = Predict(X);
		return (YPred - Y).squaredNorm();
	}

}

MatrixXd NNetwork::ConvertClassToOutput(MatrixXd& In, uint16_t NoOfLabels)
{
	MatrixXd Out = MatrixXd::Zero(In.rows(), NoOfLabels);

	for (unsigned i = 0; i < In.rows(); i++)
	{
		Out(i, (int)In(i, 0) - 1) = 1;
	}

	return Out;
}

FFeatureNormalizeOut NNetwork::FeatureNormalize(const MatrixXd &_X)
{
	FFeatureNormalizeOut Out;
	Out.mu = _X.colwise().mean();
	Out.X_norm = _X.rowwise() - Out.mu;
	Out.sigma = (Out.X_norm.colwise().squaredNorm() / (Out.X_norm.rows() - 1)).cwiseSqrt();
	Out.X_norm.array().rowwise() /= Out.sigma.array();
	return Out;
}

FPCAOut NNetwork::PCA(const Eigen::MatrixXd & _X)
{
	FPCAOut Out;
	MatrixXd sig = (_X.transpose() * _X) / _X.rows();

	JacobiSVD<MatrixXd> svd(sig, ComputeThinU);
	Out.U = svd.matrixU();
	Out.S = svd.singularValues();
	return Out;
}

MatrixXd NNetwork::RandInitializeWeights(uint64_t L_in, size_t L_out)
{
	MatrixXd Weights(L_out, L_in + 1);

	double epsilon_init = sqrt(6) / sqrt(L_in + L_out);

	for (int i = 0; i < Weights.rows(); i++)
	{
		for (int j = 0; j < Weights.cols(); j++)
		{
			Weights(i, j) = (RandomValue() * 2 - 1) * epsilon_init;
		}
	}

	return Weights;
}

MatrixXd NNetwork::Predict(MatrixXd& _X)
{
	MatrixXd Xb(_X.rows(), _X.cols() + 1);
	Xb << MatrixXd::Ones(_X.rows(), 1), _X;

	MatrixXd a2 = (Xb * Theta1.transpose()).unaryExpr(&sigmoid);

	MatrixXd a2b(a2.rows(), a2.cols() + 1);
	a2b << MatrixXd::Ones(a2.rows(), 1), a2;

	MatrixXd a3 = (a2b * Theta2.transpose()).unaryExpr(&sigmoid);

	return a3;
}

FCostFunctionOut NNetwork::nnCostFunction(const Eigen::MatrixXd& _X, const Eigen::MatrixXd& _Y, const MatrixXd& _Theta1, const MatrixXd& _Theta2, const double lambda)
{
	FCostFunctionOut Out;
	const size_t m = _X.rows();
	MatrixXd Xb(m, _X.cols() + 1);
	Xb << MatrixXd::Ones(_X.rows(), 1), _X;

	MatrixXd z2 = Xb * _Theta1.transpose();
	MatrixXd z2b(z2.rows(), z2.cols() + 1);
	z2b << MatrixXd::Ones(z2.rows(), 1), z2;

	MatrixXd a2 = z2.unaryExpr(&sigmoid);
	MatrixXd a2b(a2.rows(), a2.cols() + 1);
	a2b << MatrixXd::Ones(a2.rows(), 1), a2;

	MatrixXd a3 = (a2b * _Theta2.transpose()).unaryExpr(&sigmoid);

	Out.J = (-_Y.array() * a3.array().log() - (1 - _Y.array()) * (1 - a3.array()).log()).sum() / m
		+ lambda / (2 * m) * (
			_Theta1.rightCols(_Theta1.cols() - 1).unaryExpr(&square).sum()
			+ _Theta2.rightCols(_Theta2.cols() - 1).unaryExpr(&square).sum());

	MatrixXd Theta1Grad = MatrixXd::Zero(_Theta1.rows(), _Theta1.cols());
	MatrixXd Theta2Grad = MatrixXd::Zero(_Theta2.rows(), _Theta2.cols());

	MatrixXd delta3 = a3.array() - _Y.array();
	MatrixXd delta2 = (delta3 * _Theta2).array() * z2b.unaryExpr(&sigmoidGradient).array();

	Theta2Grad = delta3.transpose()  * a2b;
	Theta1Grad = delta2.rightCols(delta2.cols() - 1).transpose() * Xb;

	MatrixXd lamb1(_Theta1.rows(), _Theta1.cols());
	MatrixXd lamb2(_Theta2.rows(), _Theta2.cols());
	lamb1 << MatrixXd::Zero(_Theta1.rows(), 1), _Theta1.rightCols(_Theta1.cols() - 1);
	lamb2 << MatrixXd::Zero(_Theta2.rows(), 1), _Theta2.rightCols(_Theta2.cols() - 1);

	Out.Theta1_Grad = Theta1Grad * (1.0 / m) + lamb1 * (lambda / m);
	Out.Theta2_Grad = Theta2Grad * (1.0 / m) + lamb2 * (lambda / m);
	return Out;
}

Eigen::MatrixXd NNetwork::ApplyPCA(const Eigen::MatrixXd _X, const double VarianceRetained)
{
	FFeatureNormalizeOut NormOut = NNetwork::FeatureNormalize(_X);
	Mu = NormOut.mu;
	Sigma = NormOut.sigma;
	if (VarianceRetained < 1)
	{
		uint16_t k_pca = 1;
		FPCAOut PCAOut = NNetwork::PCA(NormOut.X_norm);
		for (k_pca = 1; k_pca < PCAOut.U.cols(); k_pca++)
		{
			double VarRetCalc = PCAOut.S.topRows(k_pca).sum() / PCAOut.S.sum();
			if (VarRetCalc >= VarianceRetained)
				break;
		}
		U = PCAOut.U.leftCols(k_pca);
		NormOut.X_norm *= U;
	}


	return NormOut.X_norm;
}

double NNetwork::Train(const uint64_t epoch_max, const double Learning_Rate, const double lambda)
{
	Theta1 = RandInitializeWeights(input_layer_size, hidden_layer_size);
	Theta2 = RandInitializeWeights(hidden_layer_size, num_labels);

	uint64_t epoch_count = 1;
	while (epoch_count < epoch_max)
	{
		FCostFunctionOut costOut = nnCostFunction(X, Y, Theta1, Theta2, lambda);
		Theta1 = Theta1 - Learning_Rate * costOut.Theta1_Grad * costOut.J;
		Theta2 = Theta2 - Learning_Rate *  costOut.Theta2_Grad * costOut.J;
		printf("epoch %d: %f\t\n", (int)epoch_count, costOut.J);
		epoch_count++;
	}

	return CalcTrainingError();
	//Mat<double> Weights = RandInitializeWeights(Data.TrainingSet.inputs.ColsCount(), Data.TrainingSet.outputs.ColsCount());
}

void NNetwork::Save(char* filePath)
{
	ASSERT(Theta1.rows() > 0 && Theta2.rows() > 0);
	std::ofstream file(filePath);
	if (file.is_open())
	{
		NNetwork::SavetMatrix(Mu, file);
		NNetwork::SavetMatrix(Sigma, file);
		NNetwork::SavetMatrix(U, file);
		NNetwork::SavetMatrix(Theta1, file);
		NNetwork::SavetMatrix(Theta2, file);
	}
	file.close();
}

MatrixXd NNetwork::GetMatrix(char*const filename)
{
	uint32_t Rows = 0;
	uint32_t Cols = 0;

	std::ifstream file;
	file.open(filename);
	ASSERT(!file.fail());

	while (!file.eof())
	{
		std::string line;
		std::getline(file, line);
		std::stringstream ss(line);

		std::string tempStr;
		ss >> tempStr;
		if (tempStr.compare("#") == 0)
		{
			ss >> tempStr;
			if (tempStr.compare("rows:") == 0)
			{
				ss >> Rows;
			}
			else if (tempStr.compare("columns:") == 0)
			{
				ss >> Cols;
				break;
			}
		}
	}

	MatrixXd Out(Rows, Cols);
	std::string line;
	for (uint32_t i = 0; i < Rows; i++)
	{
		std::getline(file, line);
		std::stringstream ss(line);
		for (uint16_t j = 0; j < Cols; j++)
		{
			ss >> Out(i, j);
		}
	}
	file.close();
	return Out;
}

void NNetwork::SavetMatrix(const Eigen::MatrixXd& Mat, std::ofstream&  file)
{
	ASSERT(file.is_open());
	file << "# rows: " << Mat.rows() << "\n# columns: " << Mat.cols() << std::endl;
	file << Mat << std::endl;

}

/************************************FFNetwork**************************************/
FFNetwork::FFNetwork(char * const filename)
{
	std::ifstream file;
	file.open(filename);
	ASSERT(!file.fail());

	Mu = GetMatrix(file);
	Sigma = GetMatrix(file);
	U = GetMatrix(file);
	Theta1 = GetMatrix(file).transpose();
	Theta2 = GetMatrix(file).transpose();

	file.close();
}

RowVectorXd FFNetwork::Predict(RowVectorXd& _X)
{
	RowVectorXd Xb(_X.size() + 1);
	Xb << RowVectorXd::Ones(1), _X;

	RowVectorXd a2 = (Xb * Theta1).unaryExpr(&sigmoid);

	RowVectorXd a2b(a2.size() + 1);
	a2b << RowVectorXd::Ones(1), a2;

	RowVectorXd a3 = (a2b * Theta2).unaryExpr(&sigmoid);
	return a3;
}

MatrixXd FFNetwork::Predict(MatrixXd& _XIn)
{
	// PCA Part
	MatrixXd _X = _XIn.rowwise() - Mu;
	_X.array().rowwise() /= Sigma.array();
	if (U.cols() != 0)
	{
		_X *= U;
	}

	MatrixXd Xb(_X.rows(), _X.cols() + 1);
	Xb << MatrixXd::Ones(_X.rows(), 1), _X;

	MatrixXd a2 = (Xb * Theta1).unaryExpr(&sigmoid);

	MatrixXd a2b(a2.rows(), a2.cols() + 1);
	a2b << MatrixXd::Ones(a2.rows(), 1), a2;

	MatrixXd a3 = (a2b * Theta2).unaryExpr(&sigmoid);

	return a3;
}

MatrixXd FFNetwork::GetMatrix(std::ifstream& file)
{
	uint32_t Rows = 0;
	uint32_t Cols = 0;

	ASSERT(file.is_open());

	while (!file.eof())
	{
		std::string line;
		std::getline(file, line);
		std::stringstream ss(line);

		std::string tempStr;
		ss >> tempStr;
		if (tempStr.compare("#") == 0)
		{
			ss >> tempStr;
			if (tempStr.compare("rows:") == 0)
			{
				ss >> Rows;
			}
			else if (tempStr.compare("columns:") == 0)
			{
				ss >> Cols;
				break;
			}
		}
	}

	MatrixXd Out(Rows, Cols);
	std::string line;
	for (uint32_t i = 0; i < Rows; i++)
	{
		std::getline(file, line);
		std::stringstream ss(line);
		for (uint16_t j = 0; j < Cols; j++)
		{
			ss >> Out(i, j);
		}
	}
	return Out;
}
