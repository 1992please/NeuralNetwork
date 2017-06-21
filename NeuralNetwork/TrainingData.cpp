#include "TrainingData.h"
#include <algorithm>

static char*const TrainingSetFileName = "iris_data\\iris_training.dat";
static char*const ValidationSetFileName = "iris_data\\iris_validation.dat";
static char*const TestSetFileName = "iris_data\\iris_test.dat";

TrainingData::TrainingData()
{
	UpdateDataSet(TrainingSet, TrainingSetFileName);
	UpdateDataSet(ValidationSet, ValidationSetFileName);
	UpdateDataSet(TestSet, TestSetFileName);
}

void TrainingData::UpdateDataSet(FDataSet &InDataSet, char*const filename)
{
	std::vector<std::string> lines;
	std::ifstream m_trainingDataFile;
	m_trainingDataFile.open(filename);
	while (!m_trainingDataFile.eof())
	{
		std::string line;
		std::getline(m_trainingDataFile, line);
		lines.push_back(line);
	}
	InDataSet.count = lines.size();
	InDataSet.inputs.ResizeClear(InDataSet.count, 4);
	InDataSet.outputs.ResizeClear(InDataSet.count, 3);
	InDataSet.bias.ResizeClear(InDataSet.count, 1);
	InDataSet.bias.Fill(1.0);
	for (size_t i = 0; i < lines.size(); i++)
	{
		std::stringstream ss(lines[i]);
		// this part is hardcoded depending on the dataset
		for (size_t j = 0; j < InDataSet.inputs.ColsCount(); j++)
		{
			ss >> InDataSet.inputs[i][j];
		}
		for (size_t j = 0; j < InDataSet.outputs.ColsCount(); j++)
		{
			ss >> InDataSet.outputs[i][j];
		}
	}
	InDataSet.classes = FDataSet::ConvertOuputToClass(InDataSet.outputs);
}

Mat<double> TrainingData::GetMatrix(char*const filename)
{
	uint32_t Rows;
	uint32_t Cols;

	std::ifstream m_trainingDataFile;
	m_trainingDataFile.open(filename);
	if (m_trainingDataFile.fail())
	{
		return Mat<double>();
	}
	while (!m_trainingDataFile.eof())
	{
		std::string line;
		std::getline(m_trainingDataFile, line);
		std::stringstream ss(line);

		std::string tempStr;
		ss >> tempStr;
		if (tempStr.compare("#") == 0)
		{
			ss >> tempStr;
			if (tempStr.compare("rows:") == 0)
			{
				ss >> Rows;
			}
			else if (tempStr.compare("columns:") == 0)
			{
				ss >> Cols;
				break;
			}
		}
	}

	Mat<double>Out(Rows, Cols);
	std::string line;
	for (uint32_t i = 0; i < Rows; i++)
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


Mat<double> TrainingData::GetMatrix(char*const filename, uint16_t Rows, uint16_t Cols)
{
	Mat<double>Out(Rows, Cols);
	std::ifstream m_trainingDataFile;
	m_trainingDataFile.open(filename);
	std::string line;
	for (int i = 0; i < Rows; i++)
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

std::vector<int> FDataSet::ConvertOuputToClass(Mat<double>& In)
{
	std::vector<int> out;

	for (unsigned i = 0; i < In.RowsCount(); i++)
	{
		int maxIndex = 0;
		double maxValue = In[i][maxIndex];
		for (unsigned j = 1; j < In.ColsCount(); j++)
		{
			if (In[i][j] > maxValue)
			{
				maxIndex = j;
				maxValue = In[i][maxIndex];
			}
		}
		out.push_back(maxIndex + 1);
	}
	return out;
}


Mat<double> FDataSet::ConvertClassToOutput(std::vector<int>& In)
{
	int max = 0;
	for (auto i : In)
	{
		if (i > max) max = i;
	}
	Mat<double> Out(In.size(), max, (double)0);

	for (unsigned i = 0; i < In.size(); i++)
	{
		Out[i][In[i] - 1] = 1;
	}

	return Out;
}
