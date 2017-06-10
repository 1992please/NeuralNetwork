#include "Assert.h"
#include "NeuralNet.h"
#include <algorithm>
#include "TrainingData.h"

double RandomValue(void) { return rand() / double(RAND_MAX); }

NetError EvalNetworkError(FDataSet& data, Mat<double>& Weight)
{
	NetError out;
	FeedForwardOut NetOutput = FeedForward(data.inputs, Weight, data.bias);
	out.regression_error = (data.outputs - NetOutput.Outputs).ComWiseSquared().Sum() / (data.outputs.ColsCount() * data.inputs.RowsCount());
	out.classification_error = CalcClassificationError(data.classes, FDataSet::ConvertOuputToClass(NetOutput.Outputs));
	return out;
}

FeedForwardOut FeedForward(Mat<double>& inputs, Mat<double>& weights, Mat<double>& bias)
{
	FeedForwardOut out;
	out.net = inputs.HConCat(bias) * weights;
	out.Outputs = ActivationFunc(out.net);
	return out;
}

Mat<double> InitialiseWeights(double max_weight, size_t nRows, size_t nCols)
{
	Mat<double> Weights(nRows, nCols);

	for (int i = 0; i < nRows; i++)
	{
		for (int j = 0; j < nCols; j++)
		{
			Weights[i][j] = (RandomValue() * 2 - 1) * max_weight;
		}
	}

	return Weights;
}

Mat<double> ActivationFunc(Mat<double>& In)
{
	Mat<double> out(In.RowsCount(), In.ColsCount());
	for (int i = 0; i < In.RowsCount(); i++)
	{
		for (int j = 0; j < In.ColsCount(); j++)
		{
			out[i][j] = (tanh(In[i][j]) + 1) / 2;
		}
	}
	return out;
}

Mat<double> ActivationFuncDeriv(Mat<double>& In)
{
	Mat<double> out(In.RowsCount(), In.ColsCount());
	for (int i = 0; i < In.RowsCount(); i++)
	{
		for (int j = 0; j < In.ColsCount(); j++)
		{
			const double x = tanh(In[i][j]);
			out[i][j] = (1 - x * x) / 2;
		}
	}
	return out;
}

double CalcClassificationError(std::vector<int>& class1, std::vector<int>& class2)
{
	ASSERT(class1.size() == class2.size());
	double sum = 0;
	for (int i = 0; i < class1.size(); i++)
	{
		sum += (int)(class1[i] != class2[i]);
	}

	return sum / class1.size();
}

void BackPropagation(FDataSet& DataSet, Mat<double>& Weight, double eta)
{

	size_t RandomIndex = rand() % DataSet.inputs.RowsCount();
	Mat<double> BiasedInput = DataSet.inputs.GetRow(RandomIndex).HConCat(DataSet.bias.GetRow(0));
	FeedForwardOut FFO = FeedForward(DataSet.inputs.GetRow(RandomIndex), Weight, DataSet.bias.GetRow(RandomIndex));
	Mat<double> error_vector = DataSet.outputs.GetRow(RandomIndex) - FFO.Outputs;
	Mat<double> delta = error_vector.ComWiseMul(ActivationFuncDeriv(FFO.net));
	Mat<double> WeightsDelta = BiasedInput.Transposed().KroneckerMul(delta) * eta;
	Weight = Weight + WeightsDelta;
}

TrainingOutput Train(TrainingData& Data)
{
	TrainingOutput Out;
	Mat<double> Weights = InitialiseWeights(.5, Data.TrainingSet.inputs.ColsCount() + 1, Data.TrainingSet.outputs.ColsCount());
	uint16_t epoch_count = 1;
	while (epoch_count < 2000)
	{
		BackPropagation(Data.TrainingSet, Weights, 0.1);
		NetError TrainingError = EvalNetworkError(Data.TrainingSet, Weights);
		NetError ValidationError = EvalNetworkError(Data.ValidationSet, Weights);
		NetError TestError = EvalNetworkError(Data.TestSet, Weights);


		printf("epoch: %d\tTE: %.2f\t%.2f\tVE: %.2f\t%.2f\tTE: %.2f\t%.2f\n", epoch_count,
			TrainingError.regression_error, TrainingError.classification_error,
			TestError.regression_error, TestError.classification_error,
			TestError.regression_error, TestError.classification_error);



		epoch_count++;
	}
	return Out;
}