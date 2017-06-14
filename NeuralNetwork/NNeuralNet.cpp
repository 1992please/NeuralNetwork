#include "Assert.h"
#include "NNeuralNet.h"
#include <algorithm>
#include "TrainingData.h"


static inline double RandomValue(void) { return rand() / double(RAND_MAX); }
static inline double sigmoid(const double In) { return 1.0 / (1.0 + exp(-In)); }
static inline double sigmoidGradient(const double In) { const double sig = sigmoid(In); return sig *(1 - sig); }
static inline double square(double In) { return In * In; }


NNeuralNet::NNeuralNet(TrainingData* _Data)
{
	X = _Data->TrainingSet.inputs;
	Y = _Data->TrainingSet.outputs;
	input_layer_size = X.ColsCount();
	num_labels = Y.ColsCount();
	hidden_layer_size = num_labels;
}

double NNeuralNet::CalcClassificationError(std::vector<int>& class1, std::vector<int>& class2)
{
	ASSERT(class1.size() == class2.size());
	double sum = 0;
	for (int i = 0; i < class1.size(); i++)
	{
		sum += (int)(class1[i] != class2[i]);
	}

	return sum / class1.size();
}

double NNeuralNet::CalcClassificationError(Mat<double>& Output1, Mat<double>& Output2)
{
	return CalcClassificationError(FDataSet::ConvertOuputToClass(Output1), FDataSet::ConvertOuputToClass(Output1));
}

Mat<double> NNeuralNet::ConvertClassToOutput(Mat<double>& In, uint16_t NoOfLabels)
{
	Mat<double> Out(In.RowsCount(), NoOfLabels, (double)0);

	for (unsigned i = 0; i < In.RowsCount(); i++)
	{
		Out[i][(int)In[i][0] - 1] = 1;
	}

	return Out;
}

Mat<double> NNeuralNet::RandInitializeWeights(uint16_t L_in, size_t L_out)
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

Mat<double> NNeuralNet::Predict(Mat<double>& X)
{
	Mat<double> a2 = (Mat<double>(X.RowsCount(), 1, 1.0).HConCat(X) * Theta1.Transposed()).Op(sigmoid);
	Mat<double> a3 = (Mat<double>(X.RowsCount(), 1, 1.0).HConCat(a2) * Theta2.Transposed()).Op(sigmoid);
	return a3;
}

FCostFunctionOut NNeuralNet::nnCostFunction(double lambda)
{
	FCostFunctionOut Out;
	const size_t m = X.RowsCount();

	Mat<double> z2 = Mat<double>(m, 1, 1.0).HConCat(X) * Theta1.Transposed();
	Mat<double> a2 = z2.Op(sigmoid);
	Mat<double> a3 = (Mat<double>(m, 1, 1.0).HConCat(a2) * Theta2.Transposed()).Op(sigmoid);
	Out.J = ((-Y).ComWiseMul(a3.Op(log)) - (1.0 - Y).ComWiseMul((1.0 - a3).Op(log))).Sum() / m
		+ lambda / (2 * m) *  (Theta1.GetCols(1).ComWiseSquared().Sum() + Theta2.GetCols(1).ComWiseSquared().Sum());

	Mat<double> Theta1Grad(Theta1.RowsCount(), Theta1.ColsCount(), 0.0);
	Mat<double> Theta2Grad(Theta2.RowsCount(), Theta2.ColsCount(), 0.0);

	for (size_t i = 0; i < m; i++)
	{
		Mat<double> delta3 = a3.GetRow(i) - Y.GetRow(i);
		Mat<double> delta2 = (Theta2.Transposed() * delta3.Transposed()).ComWiseMul(Mat<double>(1, 1, 1.0).HConCat(z2.GetRow(i)).Transposed().Op(sigmoidGradient));

		Theta2Grad = Theta2Grad + delta3.Transposed()  * Mat<double>(1, 1, 1.0).HConCat(a2.GetRow(i));
		Theta1Grad = Theta1Grad + delta2.GetRows(1) * Mat<double>(1, 1, 1.0).HConCat(X.GetRow(i));
	}
	Out.Theta1_Grad = Theta1Grad * (1.0 / m) + (lambda / m) * Mat<double>(Theta1.RowsCount(), 1, 0.0).HConCat(Theta1.GetCols(1));
	Out.Theta2_Grad = Theta2Grad * (1.0 / m) + (lambda / m) * Mat<double>(Theta2.RowsCount(), 1, 0.0).HConCat(Theta2.GetCols(1));



	return Out;
}


double NNeuralNet::Train(const uint32_t epoch_max, const double Learning_Rate, const double lambda)
{
	Theta1 = RandInitializeWeights(input_layer_size, hidden_layer_size);
	Theta2 = RandInitializeWeights(hidden_layer_size, num_labels);

	uint16_t epoch_count = 1;
	while (epoch_count < epoch_max)
	{
		FCostFunctionOut costOut = nnCostFunction(lambda);
		Theta1 = Theta1 - Learning_Rate * costOut.Theta1_Grad * costOut.J;
		Theta2 = Theta2 - Learning_Rate *  costOut.Theta2_Grad * costOut.J;
		printf("epoch %d: %f\t\n", epoch_count, costOut.J);
		epoch_count++;
	}

	return CalcTrainingClassificationError();
	//Mat<double> Weights = RandInitializeWeights(Data.TrainingSet.inputs.ColsCount(), Data.TrainingSet.outputs.ColsCount());
}

double NNeuralNet::CalcTrainingClassificationError()
{
	Mat<double> Y_predicted = Predict(X);
	return CalcClassificationError(FDataSet::ConvertOuputToClass(Y_predicted), FDataSet::ConvertOuputToClass(Y));
}