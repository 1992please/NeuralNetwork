#include "Assert.h"
#include "NNeuralNet.h"
#include <algorithm>
#include "TrainingData.h"


static inline double RandomValue(void) { return rand() / double(RAND_MAX); }
static inline double sigmoid(const double In) { return 1.0 / (1.0 + exp(-In)); }
static inline double sigmoidGradient(const double In) { const double sig = sigmoid(In); return sig *(1 - sig); }
static inline double square(double In) { return In * In; }


static Mat<double> ConvertClassToOutput(Mat<double>& In, uint16_t NoOfLabels)
{
	Mat<double> Out(In.RowsCount(), NoOfLabels, (double)0);

	for (unsigned i = 0; i < In.RowsCount(); i++)
	{
		Out[i][(int)In[i][0] - 1] = 1;
	}

	return Out;
}

Mat<double> RandInitializeWeights(uint16_t L_in, size_t L_out)
{
	Mat<double> Weights(L_out, L_in + 1);

	double epsilon_init = sqrt(6) / sqrt(L_in + L_out);

	for (int i = 0; i < Weights.RowsCount(); i++)
	{
		for (int j = 0; j < Weights.ColsCount(); j++)
		{
			Weights[i][j] = (RandomValue() * 2 - 1) * epsilon_init;
		}
	}

	return Weights;
}

std::vector<int> predict(Mat<double> Theta1, Mat<double> Theta2, Mat<double> X)
{
	Mat<double>(X.RowsCount(), 1, 1.0).HConCat(X).Show();
	Mat<double> a2 = (Mat<double>(X.RowsCount(), 1, 1.0).HConCat(X) * Theta1.Transposed()).Op(sigmoid);
	Mat<double> a3 = (Mat<double>(X.RowsCount(), 1, 1.0).HConCat(a2) * Theta2.Transposed()).Op(sigmoid);
	return FDataSet::ConvertOuputToClass(a3);
}

FCostFunctionOut nnCostFunction(Mat<double> Theta1, Mat<double> Theta2, uint16_t num_labels, Mat<double> X, Mat<double> Y, double lambda)
{
	FCostFunctionOut Out;

	Y = ConvertClassToOutput(Y, num_labels);

	Mat<double> z2 = Mat<double>(X.RowsCount(), 1, 1.0).HConCat(X) * Theta1.Transposed();
	Mat<double> a2 = z2.Op(sigmoid);
	Mat<double> a3 = (Mat<double>(a2.RowsCount(), 1, 1.0).HConCat(a2) * Theta2.Transposed()).Op(sigmoid);
	Out.J = ((-Y).ComWiseMul(a3.Op(log)) - (1.0 - Y).ComWiseMul((1.0 - a3).Op(log))).Sum() / X.RowsCount()
		+ lambda / (2 * X.RowsCount()) *  (Theta1.GetCols(1).ComWiseSquared().Sum() + Theta2.GetCols(1).ComWiseSquared().Sum());

	return Out;
}


void Train(TrainingData& Data)
{
	const double Weight_Max = 2;
	const double Learning_Rate = .1;
	const int epoch_max = 10000;
	//Mat<double> Weights = RandInitializeWeights(Data.TrainingSet.inputs.ColsCount(), Data.TrainingSet.outputs.ColsCount());
}
