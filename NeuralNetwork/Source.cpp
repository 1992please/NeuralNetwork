#include "TrainingData.h"
#include "NNeuralNet.h"





void main()
{

	Mat<double> R = TrainingData::GetMatrix("test_data\\R.dat");
	Mat<double> Y = TrainingData::GetMatrix("test_data\\Y.dat");
	Y.Show();
	getchar();
}

