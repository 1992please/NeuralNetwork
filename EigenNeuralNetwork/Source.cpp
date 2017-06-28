#include "NNetwork.h"
#include <iostream>

#define EIGEN_NO_DEBUG
using namespace std;

void test_train()
{
	Eigen::MatrixXd X = NNetwork::GetMatrix("data\\iris\\X.dat");
	Eigen::MatrixXd y = NNetwork::GetMatrix("data\\iris\\Y.dat");


	Eigen::MatrixXd Y =y ;
	NNetwork Net(X, Y, 1);
	cout << Net.Train(100000, .1, .1) << endl;
	Net.Save("data\\params.dat");

}

void test_load()
{
	Eigen::MatrixXd X = NNetwork::GetMatrix("data\\iris\\X.dat");
	Eigen::MatrixXd Y = NNetwork::GetMatrix("data\\iris\\y.dat");
	FFNetwork ONet("data\\params.dat");
	Eigen::MatrixXd YPred = ONet.Predict(X);
	// show results
	Eigen::MatrixXd DebugOUt(X.rows(), X.cols() + Y.cols() * 2);
	DebugOUt << X, Y, YPred;
	std::cout << std::endl << "test debug" << std::endl << DebugOUt << std::endl;
}

void test_nnCost()
{
	Eigen::MatrixXd X = NNetwork::GetMatrix("data\\cost_function_test\\X.dat");
	Eigen::MatrixXd Y = NNetwork::GetMatrix("data\\cost_function_test\\Y.dat");
	Eigen::MatrixXd Theta1 = NNetwork::GetMatrix("data\\cost_function_test\\Theta1.dat");
	Eigen::MatrixXd Theta2 = NNetwork::GetMatrix("data\\cost_function_test\\Theta2.dat");
	FCostFunctionOut Out = NNetwork::nnCostFunction(X, Y, Theta1, Theta2, 3);

	cout << Out.J << endl;
	cout << Out.Theta1_Grad << endl;
	cout << Out.Theta2_Grad << endl;
}

void main()
{
	test_load();
	//cout << "Z = \n" << NormOut.X_norm << "\nX_rec = \n" << X_rec<<"\n error \n"<<K_error<< endl;
	getchar();
}


