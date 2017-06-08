#pragma once

#include "Assert.h"

template<typename T>
struct Mat
{
private:
	T **M;
	unsigned nRows, nCols;
public:
	Mat<T>()
	{
		nRows = 0;
		nCols = 0;
		M = nullptr;
	}

	Mat<T>(const unsigned _nRows, const unsigned _nCols)
	{
		nRows = _nRows;
		nCols = _nCols;

		M = new T*[nRows];
		for (unsigned i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
		}
	}

	Mat<T>(const unsigned _nRows, const unsigned _nCols, const T& _init)
	{
		nRows = _nRows;
		nCols = _nCols;

		M = new T*[nRows];
		for (unsigned i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
			for (unsigned j = 0; j < nCols; j++)
				M[i][j] = _init;
		}
	}

	Mat<T>(unsigned Row, unsigned Col, T* arr)
	{
		nRows = Row;
		nCols = Col;

		M = new T*[nRows];
		for (unsigned i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
			for (unsigned j = 0; j < nCols; j++)
				M[i][j] = *((arr + i * Col) + j);
		}
	}

	Mat<T>(const Mat<T>& nM)
	{
		nRows = nM.nRows;
		nCols = nM.nCols;

		M = new T*[nRows];
		for (unsigned i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
			for (unsigned j = 0; j < nCols; j++)
			{
				M[i][j] = nM.M[i][j];
			}
		}
	}

	Mat<T>(Mat<T>&& nM)
	{
		nRows = nM.nRows;
		nCols = nM.nCols;
		M = nM.M;

		nM.M = nullptr;
		nM.nRows = 0;
		nM.nCols = 0;
	}

	void ResizeClear(unsigned _nRows, unsigned _nCols)
	{
		Clear();
		nRows = _nRows;
		nCols = _nCols;

		M = new T*[nRows];
		for (unsigned i = 0; i < nRows; i++)
		{
			M[i] = new T[nCols];
		}
	}

	unsigned GetRows() const
	{
		return nRows;
	}

	unsigned GetCols() const
	{
		return nCols;
	}

	~Mat()
	{
		Clear();
	}

	void Clear()
	{
		nCols = 0;
		nRows = 0;

		if (!M)
			return;

		for (unsigned i = 0; i < nRows; i++)
			delete[] M[i];

		delete[] M;
	}

	Mat<T>& Transposed()
	{
		Mat mat<T>(nCols, nRows);
		for (unsigned i = 0; i < nRows; i++)
			for (unsigned j = 0; j < nCols; j++)
				mat[j][i] = M[i][j];

		retrun mat;
	}

	inline Mat<T> operator+(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows && nCols == rhs.nCols);

		Mat<T> out(nRows, nCols);

		for (unsigned i = 0; i < nRows; i++)
			for (unsigned j = 0; j < nCols; j++)
				out[i][j] = M[i][j] + rhs[i][j];

		return out;
	}

	inline Mat<T> operator-(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows && nCols == rhs.nCols);

		Mat<T> out(nRows, nCols);

		for (unsigned i = 0; i < nRows; i++)
			for (unsigned j = 0; j < nCols; j++)
				out[i][j] = M[i][j] - rhs[i][j];

		return out;
	}

	// we need to find a way to make this faster
	inline Mat<T> operator*(const Mat& rhs)
	{
		ASSERT(nCols == rhs.nRows);

		Mat<T> out(nRows, rhs.nCols, (T)0);

		for (unsigned i = 0; i < out.nRows; i++)
			for (unsigned j = 0; j < out.nCols; j++)
				for (unsigned k = 0; k < nCols; k++)
					out[i][j] += M[i][k] * rhs[k][j];

		return out;
	}

	inline Mat<T> operator*(const T s)
	{
		Mat<T> out(nRows, nCols);

		for (unsigned i = 0; i < nRows; i++)
			for (unsigned j = 0; j < nCols; j++)
				out[i][j] = M[i][nCols] * s;

		return out;
	}

	inline Mat<T> ComWiseMul(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows && nCols == rhs.nCols);
		Mat<T> out(nRows, nCols);

		for (unsigned i = 0; i < out.nRows; i++)
			for (unsigned j = 0; j < out.nCols; j++)
				out[i][j] = M[i][j] * rhs[i][j];

		return out;
	}

	inline Mat<T> KroneckerMul(const Mat& rhs)
	{

		const unsigned p = rhs.nRows;
		const unsigned q = rhs.nCols;

		Mat<T> out(nRows * p, nCols * q);

		for (unsigned r = 0; r < nRows; r++)
			for (unsigned s = 0; s < nCols; s++)
				for (unsigned v = 0; v < rhs.nRows; v++)
					for (unsigned w = 0; w < rhs.nCols; w++)
						out[p * r + v][q * s + w] = M[r][s] * rhs[v][w];

		return out;
	}

	Mat<T> HConCat(const Mat& rhs)
	{
		ASSERT(nRows == rhs.nRows);
		Mat<T> out(nRows, nCols + rhs.nCols);

		for (unsigned i = 0; i < out.nRows; i++)
			for (unsigned j = 0; j < out.nCols; j++)
			{
				if (j < nCols)
					out[i][j] = M[i][j];
				else
					out[i][j] = rhs[i][j - nCols];
			}
		return out;
	}

	Mat<T> VConCat(const Mat& rhs)
	{
		ASSERT(nCols == rhs.nCols);
		Mat<T> out(nRows + rhs.nRows, nCols);

		for (unsigned i = 0; i < out.nRows; i++)
			for (unsigned j = 0; j < out.nCols; j++)
			{
				if (i < nRows)
					out[i][j] = M[i][j];
				else
					out[i][j] = rhs[i - nRows][j];
			}
		return out;
	}

	void Show(char * header = nullptr)
	{
		if (!header)
		{
			header = "MTemp";
		}

		printf("%s =\n", header);
		for (unsigned i = 0; i < nRows; i++)
		{
			for (unsigned j = 0; j < nCols; j++)
				printf("%.2f\t", M[i][j]);
			printf("\n");
		}
	}

	T* operator[](unsigned row) const { return M[row]; }
};

