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
	//Mat<double> Theta1 = GetMatrix("test_data\\Theta1.dat", 5, 4);
	//Mat<double> Theta2 = GetMatrix("test_data\\Theta2.dat", 3, 6);
	Mat<double> X = GetMatrix("save_data\\X.dat", 5, 3);
	Mat<double> y = GetMatrix("test_data\\y.dat", 5, 1);

	//FCostFunctionOut out = nnCostFunction(TrainingData());
	Train(TrainingData());

	getchar();
}

