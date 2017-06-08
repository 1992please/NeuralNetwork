#include "Matrix.h"
#include <algorithm>
#include "Windows.h"
#include "TrainingData.h"
#include <vector>

std::vector<int> ConvertOuputToClass(Mat<int> In)
{
	std::vector<int> out;

	for (unsigned i = 0; i < In.GetRows(); i++)
	{
		for (unsigned j = 0; j < In.GetCols(); j++)
		{
			if (In[i][j] == 1)
			{
				out.push_back(j + 1);
				break;
			}
		}
	}
	return out;
}

Mat<int> ConvertClassToOutput(std::vector<int> In)
{
	int max = 0;
	for (auto i : In)
	{
		if (i > max) max = i;
	}
	Mat<int> Out(In.size(), max, (int)0);

	for (unsigned i = 0; i < In.size(); i++)
	{
		Out[i][In[i]] = 1;
	}

	return Out;
}

void main()
{
	TrainingData data("iris_data\\iris_training.dat");

	//double x[3][2] = {
	//	{ 2, 2},
	//	{ 7, 3},
	//	{ 4, 2}
	//};

	//double y[3][2] = {
	//	{ 6, 2 },
	//	{ 5, 3},
	//	{ 7, 4 }
	//};
	//Mat<double> X(3, 2, (double*)x);
	//X.Show();
	//Mat<double> Y(3, 2, (double*)y);
	//Y.Show();
	//Mat<double> Z = X.KroneckerMul(Y);

	//Z.Show();
	//X.Show("X");
	//getchar();
}