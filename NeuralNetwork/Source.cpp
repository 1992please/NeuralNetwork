#include "TrainingData.h"
#include "NNeuralNet.h"



 Mat<double> GetMatrix(char*const filename, uint16_t Rows, uint16_t Cols)
{
	 Mat<double>Out(Rows, Cols);
	std::ifstream m_trainingDataFile;
	m_trainingDataFile.open(filename);
	std::string line;
	for(int i = 0; i < Rows; i++)
	{
		std::getline(m_trainingDataFile, line);
		std::stringstream ss(line);
		for (uint16_t j = 0; j < Cols; j++)
		{
			ss >> Out[i][j];
		}
	}
	return Out;
}

void main()
{
	printf("hello");
	Mat<double> Theta2 = GetMatrix("saved_data\\Theta2.dat", 10, 26);
	Mat<double> Theta1 = GetMatrix("saved_data\\Theta1.dat", 25, 401);
	Mat<double> X = GetMatrix("saved_data\\X.dat", 5000, 400);
	Mat<double> y = GetMatrix("saved_data\\y.dat", 5000, 1);

	FCostFunctionOut out = nnCostFunction(Theta1, Theta2, 10, X, y, 1);
	//Train(TrainingData());

	getchar();
}

