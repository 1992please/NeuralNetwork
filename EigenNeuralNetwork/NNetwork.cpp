#include "NNetwork.h"
#include "Assert.h"
#include <fstream>
#include <sstream>
#include <iostream>

static inline double RandomValue(void) { return rand() / double(RAND_MAX); }
static inline double sigmoid(const double In) { return 1.0 / (1.0 + exp(-In)); }
static inline double sigmoidGradient(const double In) { const double sig = sigmoid(In); return sig *(1 - sig); }
static inline double square(double In) { return In * In; }

NNetwork::NNetwork(MatrixXd& _X, MatrixXd&  _Y, double TestRatio)
{
	ASSERT(_X.rows() == _Y.rows());

	// shuffle the x/y matrixes

	PermutationMatrix<Dynamic, Dynamic> perm(_X.rows());
	perm.setIdentity();
	std::random_shuffle(perm.indices().data(), perm.indices().data() + perm.indices().size());
	const MatrixXd XFull = perm * _X;
	const MatrixXd YFull = perm * _Y;
	const int testSize = (int)(TestRatio * _X.rows());
	const int trainSize = (int)(_X.rows() - testSize);

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
		MatrixXd DebugOUt(Ytest.rows(), Xtest.cols() + Ytest.cols() + Ytest.cols());
		DebugOUt  << Xtest, Ytest, YPred;
		std::cout<< std::endl << "test debug" << std::endl << DebugOUt << std::endl;
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

	MatrixXd z2 = Xb * Theta1.transpose();
	MatrixXd z2b(z2.rows(), z2.cols() + 1);
	z2b << MatrixXd::Ones(z2.rows(), 1), z2;

	MatrixXd a2 = z2.unaryExpr(&sigmoid);
	MatrixXd a2b(a2.rows(), a2.cols() + 1);
	a2b << MatrixXd::Ones(a2.rows(), 1), a2;

	MatrixXd a3 = (a2b * Theta2.transpose()).unaryExpr(&sigmoid);
	return a3;
}


FCostFunctionOut NNetwork::nnCostFunction(MatrixXd& _Theta1, MatrixXd& _Theta2, double lambda)
{
	FCostFunctionOut Out;
	const size_t m = X.rows();
	MatrixXd Xb(m, X.cols() + 1);
	Xb << MatrixXd::Ones(X.rows(), 1), X;

	MatrixXd z2 = Xb * _Theta1.transpose();
	MatrixXd z2b(z2.rows(), z2.cols() + 1);
	z2b << MatrixXd::Ones(z2.rows(), 1), z2;

	MatrixXd a2 = z2.unaryExpr(&sigmoid);
	MatrixXd a2b(a2.rows(), a2.cols() + 1);
	a2b << MatrixXd::Ones(a2.rows(), 1), a2;

	MatrixXd a3 = (a2b * _Theta2.transpose()).unaryExpr(&sigmoid);

	Out.J = (-Y.array() * a3.array().log() - (1 - Y.array()) * (1 - a3.array()).log()).sum() / m
		+ lambda / (2 * m) * (
			_Theta1.rightCols(_Theta1.cols() - 1).unaryExpr(&square).sum()
			+ _Theta2.rightCols(_Theta2.cols() - 1).unaryExpr(&square).sum());

	MatrixXd Theta1Grad = MatrixXd::Zero(_Theta1.rows(), _Theta1.cols());
	MatrixXd Theta2Grad = MatrixXd::Zero(_Theta2.rows(), _Theta2.cols());

	for (size_t i = 0; i < m; i++)
	{
		MatrixXd delta3 = a3.row(i) - Y.row(i);
		MatrixXd delta2 = (_Theta2.transpose() * delta3.transpose()).array() * z2b.row(i).transpose().unaryExpr(&sigmoidGradient).array();

		Theta2Grad = Theta2Grad + delta3.transpose()  * a2b.row(i);
		Theta1Grad = Theta1Grad + delta2.bottomRows(delta2.rows() - 1) * Xb.row(i);

	}

	MatrixXd lamb1(_Theta1.rows(), _Theta1.cols());
	MatrixXd lamb2(_Theta2.rows(), _Theta2.cols());
	lamb1 << MatrixXd::Zero(_Theta1.rows(), 1), _Theta1.rightCols(_Theta1.cols() - 1);
	lamb2 << MatrixXd::Zero(_Theta2.rows(), 1), _Theta2.rightCols(_Theta2.cols() - 1);

	Out.Theta1_Grad = Theta1Grad * (1.0 / m) + lamb1 * (lambda / m);
	Out.Theta2_Grad = Theta2Grad * (1.0 / m) + lamb2 * (lambda / m);
	return Out;
}


double NNetwork::Train(const uint64_t epoch_max, const double Learning_Rate, const double lambda)
{
	Theta1 = RandInitializeWeights(input_layer_size, hidden_layer_size);
	Theta2 = RandInitializeWeights(hidden_layer_size, num_labels);

	uint64_t epoch_count = 1;
	while (epoch_count < epoch_max)
	{
		FCostFunctionOut costOut = nnCostFunction(Theta1, Theta2, lambda);
		Theta1 = Theta1 - Learning_Rate * costOut.Theta1_Grad * costOut.J;
		Theta2 = Theta2 - Learning_Rate *  costOut.Theta2_Grad * costOut.J;
		printf("epoch %d: %f\t\n", (int)epoch_count, costOut.J);
		epoch_count++;
	}

	return CalcTrainingError();
	//Mat<double> Weights = RandInitializeWeights(Data.TrainingSet.inputs.ColsCount(), Data.TrainingSet.outputs.ColsCount());
}


MatrixXd NNetwork::GetMatrix(char*const filename)
{
	uint32_t Rows;
	uint32_t Cols;

	std::ifstream m_trainingDataFile;
	m_trainingDataFile.open(filename);
	ASSERT(!m_trainingDataFile.fail())

		while (!m_trainingDataFile.eof())
		{
			std::string line;
			std::getline(m_trainingDataFile, line);
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
		std::getline(m_trainingDataFile, line);
		std::stringstream ss(line);
		for (uint16_t j = 0; j < Cols; j++)
		{
			ss >> Out(i, j);
		}
	}
	return Out;
}