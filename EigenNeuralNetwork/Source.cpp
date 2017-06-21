#include "NNetwork.h"
#include <iostream>

#define EIGEN_NO_DEBUG
using namespace std;
void main()
{
	Eigen::MatrixXd X = NNetwork::GetMatrix("data\\iris\\X.dat");
	Eigen::MatrixXd Y = NNetwork::GetMatrix("data\\iris\\Y.dat");

	NNetwork Net(X, Y);
	cout<< Net.Train(100000, .1f, 0)<< endl;

	getchar();
}


