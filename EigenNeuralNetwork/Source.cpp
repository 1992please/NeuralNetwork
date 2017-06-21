#include "NNetwork.h"
#include <iostream>

#define EIGEN_NO_DEBUG 1
using namespace std;

void test_train()
{
	Eigen::MatrixXd X = NNetwork::GetMatrix("data\\iris\\X.dat");
	Eigen::MatrixXd Y = NNetwork::GetMatrix("data\\iris\\Y.dat");

	NNetwork Net(X, Y);
	cout<< Net.Train(10000, .1f, 0)<< endl;
	Net.Save("data\\params.dat");

}

void test_load()
{
	Eigen::MatrixXd X = NNetwork::GetMatrix("data\\iris\\X.dat");
	Eigen::MatrixXd Y = NNetwork::GetMatrix("data\\iris\\Y.dat");

	NNetwork Net(X, Y);
	FFNetwork ONet("data\\params.dat");
	Eigen::MatrixXd YPred = ONet.Predict(Net.Xtest);
	// show results
	Eigen::MatrixXd DebugOUt(Net.Ytest.rows(), Net.Xtest.cols() + Net.Ytest.cols() * 2);
	DebugOUt << Net.Xtest, Net.Ytest, YPred;
	std::cout << std::endl << "test debug" << std::endl << DebugOUt << std::endl;
}

void main()
{
	test_load();

	getchar();
}


