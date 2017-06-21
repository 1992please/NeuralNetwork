#pragma once
#include "Matrix.h"


struct FCostFunctionOut
{
	double J;
	Mat<double> X_Grad;
	Mat<double> Theta_Grad;
};

class Recommender
{
public:
	Recommender();
	~Recommender();

	static FCostFunctionOut cofiCostFunc(Mat<double> Theta, Mat<double> X, Mat<double> Y, size_t num_features, double lambda);
};

