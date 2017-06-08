#include "TrainingData.h"
#include <algorithm>
TrainingData::TrainingData(char*const filename)
{
	this->filename = filename;
	UpdateDataSet();
}

void TrainingData::UpdateDataSet()
{
	FillLines();
	std::random_shuffle(lines.begin(), lines.end());
	std::random_shuffle(lines.begin(), lines.end());
	FillSets();
}

void TrainingData::FillLines()
{
	std::ifstream m_trainingDataFile;
	m_trainingDataFile.open(filename);
	while (!m_trainingDataFile.eof())
	{
		std::string line;
		std::getline(m_trainingDataFile, line);
		lines.push_back(line);
	}
}

void TrainingData::FillSets()
{
	DataSet.inputs.ResizeClear(lines.size(), 4);
	for (unsigned i = 0; i < lines.size(); i++)
	{
		std::stringstream ss(lines[i]);
		// this part is hardcoded depending on the dataset
		for (unsigned j = 0; j < DataSet.inputs.GetCols(); j++)
		{
			ss >> DataSet.inputs[i][j];
		}
		unsigned temp;
		ss >> temp;
		DataSet.bias.push_back(temp);
	}
}